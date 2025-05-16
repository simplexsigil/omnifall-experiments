import torch
import re

from typing import Tuple


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_grad_parameters = []
    skip = {}

    if cfg.NUM_GPUS > 1:
        if hasattr(model.module, "no_weight_decay"):
            skip = model.module.no_weight_decay()
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    print("Model Parameter Optimization:")
    for name_m, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for name_p, p in m.named_parameters(recurse=False):
            name = "{}.{}".format(name_m, name_p).strip(".")

            if not re.match(cfg.TRAIN.UNFREEZE_PAT, name):
                p.requires_grad = False

            if not p.requires_grad:
                no_grad_parameters.append(p)

                print(f"{name:<40} {p.requires_grad!s:>4}")
            elif is_bn:
                bn_parameters.append(p)
                print(f"{name:<40} {p.requires_grad!s:>4} BN")
            elif any(k in name for k in skip):
                zero_parameters.append(p)
                print(f"{name:<40} {p.requires_grad!s:>4} Zero WD")
            elif cfg.SOLVER.ZERO_WD_1D_PARAM and (len(p.shape) == 1 or name.endswith(".bias")):
                zero_parameters.append(p)
                print(f"{name:<40} {p.requires_grad!s:>4} Zero WD")
            else:
                non_bn_parameters.append(p)
                print(f"{name:<40} {p.requires_grad!s:>4}")

    optim_params = [
        {
            "params": bn_parameters,
            "weight_decay": cfg.BN.WEIGHT_DECAY,
            "layer_decay": 1.0,
            "apply_LARS": False,
        },
        {
            "params": non_bn_parameters,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "layer_decay": 1.0,
            "apply_LARS": cfg.SOLVER.LARS_ON,
        },
        {
            "params": zero_parameters,
            "weight_decay": 0.0,
            "layer_decay": 1.0,
            "apply_LARS": cfg.SOLVER.LARS_ON,
        },
    ]
    optim_params = [x for x in optim_params if len(x["params"])]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(bn_parameters) + len(zero_parameters) + len(
        no_grad_parameters
    ), "parameter size does not match: {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(list(model.parameters())),
    )
    print(
        "bn {}, non bn {}, zero {}, no grad {}".format(
            len(bn_parameters),
            len(non_bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
        )
    )
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"All parameters: {param_count}")
    print(f"All trainable parameters: {trainable_param_count} ({float(trainable_param_count)/param_count * 100} %)")

    if "sgd" in cfg.SOLVER.OPTIMIZING_METHOD:
        optimizer = {}

        optimizer = torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )

    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        optimizer = torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError("Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD))

    return optimizer


def get_param_groups(model, cfg):
    def _get_layer_decay(name):
        layer_id = None
        if name in ("cls_token", "mask_token"):
            layer_id = 0
        elif name.startswith("pos_embed"):
            layer_id = 0
        elif name.startswith("patch_embed"):
            layer_id = 0
        elif name.startswith("blocks"):
            layer_id = int(name.split(".")[1]) + 1
        else:
            layer_id = cfg.MVIT.DEPTH + 1
        layer_decay = cfg.SOLVER.LAYER_DECAY ** (cfg.MVIT.DEPTH + 1 - layer_id)
        return layer_id, layer_decay

    for m in model.modules():
        assert not isinstance(m, torch.nn.modules.batchnorm._NormBase), "BN is not supported with layer decay"

    non_bn_parameters_count = 0
    zero_parameters_count = 0
    no_grad_parameters_count = 0
    parameter_group_names = {}
    parameter_group_vars = {}

    skip = {}
    if cfg.NUM_GPUS > 1:
        if hasattr(model.module, "no_weight_decay"):
            skip = model.module.no_weight_decay()
            # skip = {"module." + v for v in skip}
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    for name, p in model.named_parameters():
        if not re.match(cfg.TRAIN.UNFREEZE_PAT, name):
            p.requires_grad = False

        if not p.requires_grad:
            group_name = "no_grad"
            no_grad_parameters_count += 1
            print(f"{name:<40} {p.requires_grad!s:>4}")
            continue
        name = name[len("module.") :] if name.startswith("module.") else name
        if name in skip or ((len(p.shape) == 1 or name.endswith(".bias")) and cfg.SOLVER.ZERO_WD_1D_PARAM):
            layer_id, layer_decay = _get_layer_decay(name)
            group_name = "layer_%d_%s" % (layer_id, "zero")
            weight_decay = 0.0
            zero_parameters_count += 1
            print(f"{name:<40} {p.requires_grad!s:>4} Zero WD")
        else:
            layer_id, layer_decay = _get_layer_decay(name)
            group_name = "layer_%d_%s" % (layer_id, "non_bn")
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            non_bn_parameters_count += 1
            print(f"{name:<40} {p.requires_grad!s:>4}")

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
                "layer_decay": layer_decay,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
                "layer_decay": layer_decay,
            }
        parameter_group_names[group_name]["params"].append(name)
        parameter_group_vars[group_name]["params"].append(p)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    optim_params = list(parameter_group_vars.values())

    # Check all parameters will be passed into optimizer.
    assert (
        len(list(model.parameters())) == non_bn_parameters_count + zero_parameters_count + no_grad_parameters_count
    ), "parameter size does not match: {} + {} + {} != {}".format(
        non_bn_parameters_count,
        zero_parameters_count,
        no_grad_parameters_count,
        len(list(model.parameters())),
    )
    print(
        "non bn {}, zero {}, no grad {}".format(
            non_bn_parameters_count,
            zero_parameters_count,
            no_grad_parameters_count,
        )
    )

    return optim_params


def generate_latex_rows(data_dict):
    # Define the order of datasets to match the example
    dataset_order = [
        "cmdfall",
        "up_fall",  # Note: The data might have "up" or "up_fall"
        "le2i",
        "gmdcsa24",
        "edf",
        "occu",
        "caucafall",
        "mcfd",
        "combined",
        "OOPS",  # Added Oops-Fall to the list
    ]

    # Dataset name mappings for citation with proper width padding
    dataset_citations = {
        "cmdfall": "CMDFall~\\cite{cmdfall}",
        "up_fall": "UP-Fall~\\cite{up-fall}",
        "le2i": "Le2i~\\cite{le2i}",
        "gmdcsa24": "GMDCSA24~\\cite{gmdcsa}",
        "edf": "EDF~\\cite{edf-occu}",
        "occu": "OCCU~\\cite{edf-occu}",
        "caucafall": "CaucaFall~\\cite{cauca}",
        "mcfd": "MCFD~\\cite{mcfd}",
        "combined": "\\textbf{Overall}",
        "OOPS": "\\textbf{Oops-Fall}",
    }

    # Process each dataset in the specified order
    for dataset in dataset_order:
        # Get dataset name with citation
        dataset_name = dataset_citations.get(dataset, dataset)

        dataset_dict = data_dict.get(dataset, {})
        # Extract metrics
        bal_acc = dataset_dict.get(f"eval_balanced_accuracy", "")
        acc = dataset_dict.get(f"eval_accuracy", "")
        f1 = dataset_dict.get(f"eval_macro_f1", "")

        fall_se = dataset_dict.get(f"eval_fall_sensitivity", "")
        fall_sp = dataset_dict.get(f"eval_fall_specificity", "")
        fall_f1 = dataset_dict.get(f"eval_fall_f1", "")

        fallen_se = dataset_dict.get(f"eval_fallen_sensitivity", "")
        fallen_sp = dataset_dict.get(f"eval_fallen_specificity", "")
        fallen_f1 = dataset_dict.get(f"eval_fallen_f1", "")

        union_se = dataset_dict.get(f"eval_fall_union_fallen_sensitivity", "")
        union_sp = dataset_dict.get(f"eval_fall_union_fallen_specificity", "")
        union_f1 = dataset_dict.get(f"eval_fall_union_fallen_f1", "")

        # Format values to 2 decimal places
        def format_val(val):
            if val == "":
                return "$\\sim$"
            return f"{float(val):.2f}"

        # Generate LaTeX row with proper alignment
        row = f"{dataset_name:<22} & {format_val(bal_acc)} & {format_val(acc)} & {format_val(f1)} && "
        row += f"{format_val(fall_se)} & {format_val(fall_sp)} & {format_val(fall_f1)} && "
        row += f"{format_val(fallen_se)} & {format_val(fallen_sp)} & {format_val(fallen_f1)} && "
        row += f"{format_val(union_se)} & {format_val(union_sp)} & {format_val(union_f1)} \\\\"

        print(row)

        # Add midrule after Combined/Overall
        if dataset == "combined":
            print("\\midrule")


if __name__ == "__main__":
    data_dict = {
        "caucafall_eval_accuracy": "0.46809",
        "caucafall_eval_balanced_accuracy": "0.35574",
        "caucafall_eval_dist_fall": "0.19149",
        "caucafall_eval_dist_fallen": "0.19149",
        "caucafall_eval_dist_other": "0.12766",
        "caucafall_eval_dist_sit_down": "0.04255",
        "caucafall_eval_dist_sitting": "0.08511",
        "caucafall_eval_dist_stand_up": "0.06383",
        "caucafall_eval_dist_standing": "0.06383",
        "caucafall_eval_dist_walk": "0.23404",
        "caucafall_eval_fall": "0.63636",
        "caucafall_eval_fall_f1": "0.63636",
        "caucafall_eval_fall_sensitivity": "0.77778",
        "caucafall_eval_fall_specificity": "0.84211",
        "caucafall_eval_fall_union_fallen_f1": "0.76471",
        "caucafall_eval_fall_union_fallen_sensitivity": "0.72222",
        "caucafall_eval_fall_union_fallen_specificity": "0.89655",
        "caucafall_eval_fallen": "0.5",
        "caucafall_eval_fallen_f1": "0.5",
        "caucafall_eval_fallen_sensitivity": "0.33333",
        "caucafall_eval_fallen_specificity": "1",
        "caucafall_eval_loss": "1.89622",
        "caucafall_eval_macro_f1": "0.30611",
        "caucafall_eval_runtime": "11.5361",
        "caucafall_eval_sample_count": "47",
        "caucafall_eval_samples_per_second": "4.074",
        "caucafall_eval_sit_down": "0",
        "caucafall_eval_sitting": "0.25",
        "caucafall_eval_stand_up": "0",
        "caucafall_eval_steps_per_second": "0.087",
        "caucafall_eval_walk": "0.5625",
        "caucafall_test_size": "47",
        "caucafall_train_size": "176",
        "caucafall_val_size": "25",
        "cmdfall_epoch": "0.2457",
        "cmdfall_eval_accuracy": "0.49738",
        "cmdfall_eval_balanced_accuracy": "0.35288",
        "cmdfall_eval_dist_fall": "0.13193",
        "cmdfall_eval_dist_fallen": "0.11098",
        "cmdfall_eval_dist_lie_down": "0.04519",
        "cmdfall_eval_dist_lying": "0.02612",
        "cmdfall_eval_dist_other": "0.11121",
        "cmdfall_eval_dist_sit_down": "0.07126",
        "cmdfall_eval_dist_sitting": "0.04451",
        "cmdfall_eval_dist_stand_up": "0.20398",
        "cmdfall_eval_dist_standing": "0.04974",
        "cmdfall_eval_dist_walk": "0.20507",
        "cmdfall_eval_fall": "0.50624",
        "cmdfall_eval_fall_f1": "0.50624",
        "cmdfall_eval_fall_sensitivity": "0.53408",
        "cmdfall_eval_fall_specificity": "0.91247",
        "cmdfall_eval_fall_union_fallen_f1": "0.63302",
        "cmdfall_eval_fall_union_fallen_sensitivity": "0.68322",
        "cmdfall_eval_fall_union_fallen_specificity": "0.84747",
        "cmdfall_eval_fallen": "0.69921",
        "cmdfall_eval_fallen_f1": "0.69921",
        "cmdfall_eval_fallen_sensitivity": "0.77487",
        "cmdfall_eval_fallen_specificity": "0.94488",
        "cmdfall_eval_lie_down": "0.10375",
        "cmdfall_eval_loss": "1.62511",
        "cmdfall_eval_lying": "0",
        "cmdfall_eval_macro_f1": "0.34222",
        "cmdfall_eval_other": "0.25098",
        "cmdfall_eval_runtime": "125.6421",
        "cmdfall_eval_sample_count": "17570",
        "cmdfall_eval_samples_per_second": "139.842",
        "cmdfall_eval_sit_down": "0.23652",
        "cmdfall_eval_sitting": "0.35294",
        "cmdfall_eval_stand_up": "0.49908",
        "cmdfall_eval_standing": "0.15909",
        "cmdfall_eval_steps_per_second": "2.189",
        "cmdfall_eval_walk": "0.61437",
        "cmdfall_test_size": "17570",
        "cmdfall_train_size": "20884",
        "cmdfall_val_size": "3689",
        "combined_epoch": "0.2457",
        "combined_eval_accuracy": "0.49542",
        "combined_eval_balanced_accuracy": "0.3526",
        "combined_eval_dist_fall": "0.13229",
        "combined_eval_dist_fallen": "0.11294",
        "combined_eval_dist_lie_down": "0.04421",
        "combined_eval_dist_lying": "0.02552",
        "combined_eval_dist_other": "0.11145",
        "combined_eval_dist_sit_down": "0.07077",
        "combined_eval_dist_sitting": "0.04619",
        "combined_eval_dist_stand_up": "0.20003",
        "combined_eval_dist_standing": "0.05143",
        "combined_eval_dist_walk": "0.20516",
        "combined_eval_fall": "0.49286",
        "combined_eval_fall_f1": "0.49286",
        "combined_eval_fall_sensitivity": "0.5175",
        "combined_eval_fall_specificity": "0.91119",
        "combined_eval_fall_union_fallen_f1": "0.62749",
        "combined_eval_fall_union_fallen_sensitivity": "0.67318",
        "combined_eval_fall_union_fallen_specificity": "0.84649",
        "combined_eval_fallen": "0.71061",
        "combined_eval_fallen_f1": "0.71061",
        "combined_eval_fallen_sensitivity": "0.78136",
        "combined_eval_fallen_specificity": "0.94681",
        "combined_eval_lie_down": "0.11159",
        "combined_eval_loss": "1.62737",
        "combined_eval_lying": "0.00431",
        "combined_eval_macro_f1": "0.34428",
        "combined_eval_other": "0.23808",
        "combined_eval_runtime": "127.4475",
        "combined_eval_sample_count": "18142",
        "combined_eval_samples_per_second": "142.349",
        "combined_eval_sit_down": "0.21969",
        "combined_eval_sitting": "0.35189",
        "combined_eval_stand_up": "0.49716",
        "combined_eval_standing": "0.20466",
        "combined_eval_steps_per_second": "2.228",
        "combined_eval_walk": "0.61201",
        "edf_epoch": "0.2457",
        "edf_eval_accuracy": "0.38281",
        "edf_eval_balanced_accuracy": "0.21534",
        "edf_eval_dist_fall": "0.125",
        "edf_eval_dist_fallen": "0.15625",
        "edf_eval_dist_lie_down": "0.01562",
        "edf_eval_dist_lying": "0.01562",
        "edf_eval_dist_other": "0.09375",
        "edf_eval_dist_sit_down": "0.01562",
        "edf_eval_dist_sitting": "0.01562",
        "edf_eval_dist_stand_up": "0.17188",
        "edf_eval_dist_standing": "0.23438",
        "edf_eval_dist_walk": "0.15625",
        "edf_eval_fall": "0.3125",
        "edf_eval_fall_f1": "0.3125",
        "edf_eval_fall_sensitivity": "0.3125",
        "edf_eval_fall_specificity": "0.90179",
        "edf_eval_fall_union_fallen_f1": "0.62791",
        "edf_eval_fall_union_fallen_sensitivity": "0.75",
        "edf_eval_fall_union_fallen_specificity": "0.75",
        "edf_eval_fallen": "0.48148",
        "edf_eval_fallen_f1": "0.48148",
        "edf_eval_fallen_sensitivity": "0.65",
        "edf_eval_fallen_specificity": "0.80556",
        "edf_eval_lie_down": "0",
        "edf_eval_loss": "2.06345",
        "edf_eval_lying": "0",
        "edf_eval_macro_f1": "0.18669",
        "edf_eval_other": "0",
        "edf_eval_runtime": "12.7871",
        "edf_eval_sample_count": "128",
        "edf_eval_samples_per_second": "10.01",
        "edf_eval_sit_down": "0",
        "edf_eval_sitting": "0",
        "edf_eval_stand_up": "0.13333",
        "edf_eval_standing": "0.51852",
        "edf_eval_steps_per_second": "0.156",
        "edf_eval_walk": "0.42105",
        "edf_test_size": "128",
        "edf_train_size": "302",
        "edf_val_size": "78",
        "eval/accuracy": "0.36634",
        "eval/balanced_accuracy": "0.29213",
        "eval/dist_fall": "0.15842",
        "eval/dist_fallen": "0.15842",
        "eval/dist_lie_down": "0.03226",
        "eval/dist_lying": "0.04301",
        "eval/dist_other": "0.07921",
        "eval/dist_sit_down": "0.0396",
        "eval/dist_sitting": "0.0396",
        "eval/dist_stand_up": "0.19802",
        "eval/dist_standing": "0.26733",
        "eval/dist_walk": "0.05941",
        "eval/fall": "0.33333",
        "eval/fall_f1": "0.33333",
        "eval/fall_sensitivity": "0.25",
        "eval/fall_specificity": "0.95294",
        "eval/fall_union_fallen_f1": "0.49123",
        "eval/fall_union_fallen_sensitivity": "0.4375",
        "eval/fall_union_fallen_specificity": "0.84058",
        "eval/fallen": "0.48485",
        "eval/fallen_f1": "0.48485",
        "eval/fallen_sensitivity": "0.5",
        "eval/fallen_specificity": "0.89412",
        "eval/lie_down": "0",
        "eval/loss": "1.85609",
        "eval/lying": "0",
        "eval/macro_f1": "0.24231",
        "eval/other": "0.33333",
        "eval/runtime": "10.9957",
        "eval/sample_count": "101",
        "eval/samples_per_second": "9.185",
        "eval/sit_down": "0",
        "eval/sitting": "0",
        "eval/stand_up": "0",
        "eval/standing": "0",
        "eval/steps_per_second": "0.182",
        "eval/walk": "0.27586",
        "gmdcsa24_epoch": "0.2457",
        "gmdcsa24_eval_accuracy": "0.3871",
        "gmdcsa24_eval_balanced_accuracy": "0.22822",
        "gmdcsa24_eval_dist_fall": "0.1828",
        "gmdcsa24_eval_dist_fallen": "0.1828",
        "gmdcsa24_eval_dist_lie_down": "0.03226",
        "gmdcsa24_eval_dist_lying": "0.04301",
        "gmdcsa24_eval_dist_other": "0.15054",
        "gmdcsa24_eval_dist_sit_down": "0.04301",
        "gmdcsa24_eval_dist_sitting": "0.1828",
        "gmdcsa24_eval_dist_stand_up": "0.02151",
        "gmdcsa24_eval_dist_standing": "0.02151",
        "gmdcsa24_eval_dist_walk": "0.13978",
        "gmdcsa24_eval_fall": "0.5",
        "gmdcsa24_eval_fall_f1": "0.5",
        "gmdcsa24_eval_fall_sensitivity": "0.35294",
        "gmdcsa24_eval_fall_specificity": "0.98684",
        "gmdcsa24_eval_fall_union_fallen_f1": "0.65672",
        "gmdcsa24_eval_fall_union_fallen_sensitivity": "0.64706",
        "gmdcsa24_eval_fall_union_fallen_specificity": "0.81356",
        "gmdcsa24_eval_fallen": "0.69767",
        "gmdcsa24_eval_fallen_f1": "0.69767",
        "gmdcsa24_eval_fallen_sensitivity": "0.88235",
        "gmdcsa24_eval_fallen_specificity": "0.85526",
        "gmdcsa24_eval_lie_down": "0",
        "gmdcsa24_eval_loss": "1.87017",
        "gmdcsa24_eval_lying": "0",
        "gmdcsa24_eval_macro_f1": "0.22425",
        "gmdcsa24_eval_other": "0.33333",
        "gmdcsa24_eval_runtime": "10.5111",
        "gmdcsa24_eval_sample_count": "93",
        "gmdcsa24_eval_samples_per_second": "8.848",
        "gmdcsa24_eval_sit_down": "0",
        "gmdcsa24_eval_sitting": "0.32258",
        "gmdcsa24_eval_stand_up": "0",
        "gmdcsa24_eval_standing": "0",
        "gmdcsa24_eval_steps_per_second": "0.19",
        "gmdcsa24_eval_walk": "0.38889",
        "gmdcsa24_test_size": "93",
        "gmdcsa24_train_size": "213",
        "gmdcsa24_val_size": "152",
        "le2i_epoch": "0.2457",
        "le2i_eval_accuracy": "0.48768",
        "le2i_eval_balanced_accuracy": "0.34736",
        "le2i_eval_dist_fall": "0.10837",
        "le2i_eval_dist_fallen": "0.10345",
        "le2i_eval_dist_other": "0.14778",
        "le2i_eval_dist_sit_down": "0.06897",
        "le2i_eval_dist_sitting": "0.133",
        "le2i_eval_dist_stand_up": "0.10345",
        "le2i_eval_dist_standing": "0.01478",
        "le2i_eval_dist_walk": "0.3202",
        "le2i_eval_fall": "0.4375",
        "le2i_eval_fall_f1": "0.4375",
        "le2i_eval_fall_sensitivity": "0.63636",
        "le2i_eval_fall_specificity": "0.8453",
        "le2i_eval_fall_union_fallen_f1": "0.61947",
        "le2i_eval_fall_union_fallen_sensitivity": "0.81395",
        "le2i_eval_fall_union_fallen_specificity": "0.78125",
        "le2i_eval_fallen": "0.77551",
        "le2i_eval_fallen_f1": "0.77551",
        "le2i_eval_fallen_sensitivity": "0.90476",
        "le2i_eval_fallen_specificity": "0.95055",
        "le2i_eval_loss": "1.65173",
        "le2i_eval_macro_f1": "0.30218",
        "le2i_eval_runtime": "10.5861",
        "le2i_eval_sample_count": "203",
        "le2i_eval_samples_per_second": "19.176",
        "le2i_eval_sit_down": "0",
        "le2i_eval_sitting": "0.07143",
        "le2i_eval_stand_up": "0.19048",
        "le2i_eval_steps_per_second": "0.378",
        "le2i_eval_walk": "0.66667",
        "le2i_test_size": "203",
        "le2i_train_size": "670",
        "le2i_val_size": "94",
        "mcfd_test_size": "0",
        "mcfd_train_size": "1352",
        "mcfd_val_size": "0",
        "occu_epoch": "0.2457",
        "occu_eval_accuracy": "0.36634",
        "occu_eval_balanced_accuracy": "0.29213",
        "occu_eval_dist_fall": "0.15842",
        "occu_eval_dist_fallen": "0.15842",
        "occu_eval_dist_other": "0.07921",
        "occu_eval_dist_sit_down": "0.0396",
        "occu_eval_dist_sitting": "0.0396",
        "occu_eval_dist_stand_up": "0.19802",
        "occu_eval_dist_standing": "0.26733",
        "occu_eval_dist_walk": "0.05941",
        "occu_eval_fall": "0.33333",
        "occu_eval_fall_f1": "0.33333",
        "occu_eval_fall_sensitivity": "0.25",
        "occu_eval_fall_specificity": "0.95294",
        "occu_eval_fall_union_fallen_f1": "0.49123",
        "occu_eval_fall_union_fallen_sensitivity": "0.4375",
        "occu_eval_fall_union_fallen_specificity": "0.84058",
        "occu_eval_fallen": "0.48485",
        "occu_eval_fallen_f1": "0.48485",
        "occu_eval_fallen_sensitivity": "0.5",
        "occu_eval_fallen_specificity": "0.89412",
        "occu_eval_loss": "1.85609",
        "occu_eval_macro_f1": "0.24231",
        "occu_eval_runtime": "10.9957",
        "occu_eval_sample_count": "101",
        "occu_eval_samples_per_second": "9.185",
        "occu_eval_sit_down": "0",
        "occu_eval_sitting": "0",
        "occu_eval_stand_up": "0",
        "occu_eval_steps_per_second": "0.182",
        "occu_eval_walk": "0.27586",
        "occu_test_size": "101",
        "occu_train_size": "289",
        "occu_val_size": "94",
        "train/epoch": "0.2457",
        "train/global_step": "100",
        "train/grad_norm": "9.22721",
        "train/learning_rate": "8e-05",
        "train/loss": "1.6787",
        "up_fall_train_size": "2150",
        "up_fall_val_size": "140",
    }

    # Call the function to generate the LaTeX rows
    generate_latex_rows(data_dict)
