from data.feature_dataset import FeatureVideoDataset, MultiFeatureDataset


def main():
    import argparse
    import os
    import sys
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    import pandas as pd
    import seaborn as sns

    label2idx = {
        "Walk": 0,
        "Fall": 1,
        "Fallen": 2,
        "Sit Down": 3,
        "Sitting": 4,
        "Lie Down": 5,
        "Lying": 6,
        "Stand Up": 7,
        "Standing": 8,
        "Other": 9,
        "no_fall": -1,
    }

    idx2label = {v: k for k, v in label2idx.items()}

    parser = argparse.ArgumentParser(description="Test Feature Video Dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--visualize", action="store_true", help="Visualize feature statistics")
    parser.add_argument(
        "--save_dir", type=str, default="./outputs/dataset_statistics", help="Directory to save visualizations"
    )

    args = parser.parse_args()
    frames_per_feature = 16

    omnifall_root = "/lsdf/data/activity/fall_detection/cvhci_fall"
    annotations_file_cauca = "labels/dataset/caucafall.csv"
    feature_root_cauca = "caucafall/features/i3d"
    feature_fps_cauca = 20.0
    feature_stride_cauca = 1 / feature_fps_cauca

    feature_root_cmdfall = "cmdfall/features/i3d"
    annotations_file_cmdfall = "labels/dataset/cmdfall.csv"
    feature_fps_cmdfall = 20.0
    feature_stride_cmdfall = 1 / feature_fps_cmdfall

    feature_root_edf = "edf/features/i3d"
    annotations_file_edf = "labels/dataset/edf.csv"
    feature_fps_edf = 30.0
    feature_stride_edf = 1 / feature_fps_edf

    feature_root_GMDCSA24 = "GMDCSA24/features/i3d"
    annotations_file_GMDCSA24 = "labels/dataset/GMDCSA24.csv"
    feature_fps_GMDCSA24 = 30.0
    feature_stride_GMDCSA24 = 1 / feature_fps_GMDCSA24

    feature_root_le2i = "le2i/features/i3d"
    annotations_file_le2i = "labels/dataset/le2i.csv"
    feature_fps_le2i = 25.0
    feature_stride_le2i = 1 / feature_fps_le2i

    feature_root_mcfd = "mcfd/features/i3d"
    annotations_file_mcfd = "labels/dataset/mcfd.csv"
    feature_fps_mcfd = 30.0
    feature_stride_mcfd = 1 / feature_fps_mcfd

    feature_root_occu = "occu/features/i3d"
    annotations_file_occu = "labels/dataset/occu.csv"
    feature_fps_occu = 30.0
    feature_stride_occu = 1 / feature_fps_occu

    feature_root_up_fall = "up_fall/features/i3d"
    annotations_file_up_fall = "labels/dataset/up_fall.csv"
    feature_fps_up_fall = 18.0
    feature_stride_up_fall = 1 / feature_fps_up_fall

    get_features = False

    datasets = [
        FeatureVideoDataset(
            feature_root=os.path.join(omnifall_root, feature_root_cauca),
            annotations_file=os.path.join(omnifall_root, annotations_file_cauca),
            feature_fps=feature_fps_cauca,
            feature_frames=16,
            feature_stride=feature_stride_cauca,
            feature_type="i3d",
            feature_ext=".h5",
            mode="all",
            dataset_name="CAUCAFall",
            get_features=get_features,
        ),
        FeatureVideoDataset(
            feature_root=os.path.join(omnifall_root, feature_root_cmdfall),
            annotations_file=os.path.join(omnifall_root, annotations_file_cmdfall),
            feature_fps=feature_fps_cmdfall,
            feature_frames=16,
            feature_stride=feature_stride_cmdfall,
            feature_type="i3d",
            feature_ext=".h5",
            mode="all",
            dataset_name="CMDFALL",
            get_features=get_features,
        ),
        FeatureVideoDataset(
            feature_root=os.path.join(omnifall_root, feature_root_edf),
            annotations_file=os.path.join(omnifall_root, annotations_file_edf),
            feature_fps=feature_fps_edf,
            feature_frames=16,
            feature_stride=feature_stride_edf,
            feature_type="i3d",
            feature_ext=".h5",
            mode="all",
            dataset_name="EDF",
            get_features=get_features,
        ),
        FeatureVideoDataset(
            feature_root=os.path.join(omnifall_root, feature_root_GMDCSA24),
            annotations_file=os.path.join(omnifall_root, annotations_file_GMDCSA24),
            feature_fps=feature_fps_GMDCSA24,
            feature_frames=16,
            feature_stride=feature_stride_GMDCSA24,
            feature_type="i3d",
            feature_ext=".h5",
            mode="all",
            dataset_name="GMDCSA24",
            get_features=get_features,
        ),
        FeatureVideoDataset(
            feature_root=os.path.join(omnifall_root, feature_root_le2i),
            annotations_file=os.path.join(omnifall_root, annotations_file_le2i),
            feature_fps=feature_fps_le2i,
            feature_frames=16,
            feature_stride=feature_stride_le2i,
            feature_type="i3d",
            feature_ext=".h5",
            mode="all",
            dataset_name="Le2i",
            get_features=get_features,
        ),
        FeatureVideoDataset(
            feature_root=os.path.join(omnifall_root, feature_root_mcfd),
            annotations_file=os.path.join(omnifall_root, annotations_file_mcfd),
            feature_fps=feature_fps_mcfd,
            feature_frames=16,
            feature_stride=feature_stride_mcfd,
            feature_type="i3d",
            feature_ext=".h5",
            mode="all",
            dataset_name="MCFD",
            get_features=get_features,
        ),
        FeatureVideoDataset(
            feature_root=os.path.join(omnifall_root, feature_root_occu),
            annotations_file=os.path.join(omnifall_root, annotations_file_occu),
            feature_fps=feature_fps_occu,
            feature_frames=16,
            feature_stride=feature_stride_occu,
            feature_type="i3d",
            feature_ext=".h5",
            mode="all",
            dataset_name="OCCU",
            get_features=get_features,
        ),
        FeatureVideoDataset(
            feature_root=os.path.join(omnifall_root, feature_root_up_fall),
            annotations_file=os.path.join(omnifall_root, annotations_file_up_fall),
            feature_fps=feature_fps_up_fall,
            feature_frames=16,
            feature_stride=feature_stride_up_fall,
            feature_type="i3d",
            feature_ext=".h5",
            mode="all",
            dataset_name="UP-Fall",
            get_features=get_features,
        ),
    ]

    multi_dataset = MultiFeatureDataset(datasets)
    print(f"MultiFeatureDataset created with {len(multi_dataset)} segments")

    if len(multi_dataset) == 0:
        print("MultiFeatureDataset is empty. No statistics to generate.")
        return

    # Create DataLoader
    dataloader = DataLoader(
        multi_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"DataLoader created with {len(dataloader)} batches")

    # Create save directory
    save_dir_path = Path(args.save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {save_dir_path.resolve()}")

    # Collect data from DataLoader
    all_dataset_names = []
    all_labels = []
    all_start_times = []
    all_end_times = []
    all_video_ids = []
    all_cameras = []  # New list for camera information

    print("Collecting data from DataLoader...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        all_dataset_names.extend(batch["dataset_name"])
        all_labels.extend(batch["label"].cpu().numpy())
        all_start_times.extend(batch["start"].cpu().numpy())  # Assuming 'start' and 'end' are the keys
        all_end_times.extend(batch["end"].cpu().numpy())  # for start and end times
        all_video_ids.extend(batch["video_id"])
        if "camera" in batch:  # Check if camera info is available
            all_cameras.extend(batch["camera"])
        else:  # Add placeholder if camera info is not in batch
            all_cameras.extend([None] * len(batch["dataset_name"]))

    # Create Pandas DataFrame
    df_data = {
        "dataset_name": all_dataset_names,
        "label": all_labels,
        "start_time": all_start_times,
        "end_time": all_end_times,
        "video_id": all_video_ids,
        "camera": all_cameras,  # Add camera to DataFrame
    }
    df = pd.DataFrame(df_data)
    df["duration"] = df["end_time"] - df["start_time"]
    df["label_name"] = df["label"].map(idx2label)  # CRITICAL: Ensure idx2label output matches color_group_map keys

    print(f"Collected {len(df)} segments.")
    if df.empty:
        print("No data collected. Skipping plot generation.")
        return

    # --- Define Color Scheme ---
    actual_label_names_from_data = sorted(
        list(df["label_name"].dropna().unique())
    )  # Use dropna() to avoid issues with None
    print(f"DEBUG: Unique label names found in data (used for color mapping): {actual_label_names_from_data}")

    color_group_map = {
        "Fall": "coral",
        "Fallen": "firebrick",  # Orange group
        "Lie Down": "palegreen",
        "Lying": "mediumseagreen",
        "Stand Up": "forestgreen",  # Green group
        "Sit Down": "khaki",
        "Sitting": "darkkhaki",  # Yellow group
        "Walk": "skyblue",
        "Standing": "steelblue",
        "Other": "darkorchid",
        "NonFall": "cornflowerblue",  # Blue group
    }
    default_color = "pink"

    label_color_map = {label: color_group_map.get(label, default_color) for label in actual_label_names_from_data}
    print(f"DEBUG: Generated label_color_map: {label_color_map}")

    # Define a preferred order for labels in plots.
    preferred_order = [
        "Fall",
        "Fallen",
        "Lie Down",
        "Lying",
        "Stand Up",
        "Sit Down",
        "Sitting",
        "Walk",
        "Standing",
        "NonFall",
        "Other",
    ]

    # Construct plot_label_order: labels from preferred_order present in data, then other data labels.
    plot_label_order = [label for label in preferred_order if label in actual_label_names_from_data]
    for label in actual_label_names_from_data:
        if label not in plot_label_order:
            plot_label_order.append(label)

    # If plot_label_order ended up empty (e.g., no overlap), fallback to actual_label_names_from_data
    if not plot_label_order and actual_label_names_from_data:
        plot_label_order = actual_label_names_from_data
    elif not actual_label_names_from_data:  # If there are no labels in data
        plot_label_order = []

    print(f"DEBUG: plot_label_order for plots: {plot_label_order}")

    # --- Calculate Corrected Total Duration and Dataset Order ---
    print("Calculating total annotated durations and dataset order...")
    camera_filter_datasets = {"EDF", "OCCU", "MCFD", "CMDFALL", "UP-Fall"}
    dataset_total_durations = {}

    for dataset_name in df["dataset_name"].unique():
        dataset_df = df[df["dataset_name"] == dataset_name].copy()
        if dataset_name in camera_filter_datasets:
            if "camera" not in dataset_df.columns or dataset_df["camera"].isnull().all():
                print(
                    f"Warning: Camera information missing for {dataset_name}, using all segments for duration calculation."
                )
                dataset_total_durations[dataset_name] = dataset_df["duration"].sum()
                continue

            # Ensure 'camera' column is not all None before proceeding with filtering
            if dataset_df["camera"].notna().any():
                # Pick the first non-null camera for this entire dataset
                first_camera_series = dataset_df["camera"].dropna()
                if not first_camera_series.empty:
                    chosen_camera = first_camera_series.iloc[0]
                    print(f"Info: For dataset {dataset_name}, using camera '{chosen_camera}' for duration calculation.")
                    filtered_df = dataset_df[dataset_df["camera"] == chosen_camera]
                    dataset_total_durations[dataset_name] = filtered_df["duration"].sum()
                else:
                    # This case should ideally not be reached if dataset_df["camera"].notna().any() is true,
                    # but as a fallback, sum all durations if somehow all cameras became null after initial check.
                    print(
                        f"Warning: No non-null cameras found for {dataset_name} after initial check, using all segments."
                    )
                    dataset_total_durations[dataset_name] = dataset_df["duration"].sum()
            else:  # If all camera entries are None for this dataset
                dataset_total_durations[dataset_name] = dataset_df["duration"].sum()
        else:
            dataset_total_durations[dataset_name] = dataset_df["duration"].sum()

    # Sort datasets by total duration (descending)
    sorted_dataset_names = sorted(
        dataset_total_durations.keys(), key=lambda k: dataset_total_durations[k], reverse=True
    )

    total_duration_series = pd.Series(dataset_total_durations).reindex(sorted_dataset_names)

    # --- Generate and Save Plots ---
    print("Generating and saving plots...")

    # Plot 1: Total Annotated Segment Duration per Dataset (Stacked Horizontal Bar Plot, Log Scale)
    if plot_label_order:  # Proceed only if there are labels to plot

        duration_per_dataset_label = df.groupby(["dataset_name", "label_name"])["duration"].sum().unstack(fill_value=0)

        for col in plot_label_order:
            if col not in duration_per_dataset_label:
                duration_per_dataset_label[col] = 0

        duration_per_dataset_label = duration_per_dataset_label.reindex(index=sorted_dataset_names).fillna(0)
        duration_per_dataset_label = duration_per_dataset_label[plot_label_order].fillna(0)

        duration_per_dataset_label_plot = duration_per_dataset_label.loc[
            :, (duration_per_dataset_label != 0).any(axis=0)
        ]
        # --- End of data preparation ---

        if not duration_per_dataset_label_plot.empty:
            # print("\nDEBUG: Data for Plots (Aggregated Durations - Head):")
            # print(duration_per_dataset_label_plot.head())

            total_durations = total_duration_series
            # Use a small threshold for "positive" for log scale to avoid issues with true zeros
            # and to ensure very small durations are still plotted.
            total_durations_positive = total_durations[total_durations > 1e-9].sort_values(ascending=True)

            if total_durations_positive.empty:
                print("Skipping plot: All datasets have near-zero or zero total duration after filtering.")
            else:
                # Reindex original data based on sorted positive totals for consistent row order
                df_proportions_input = duration_per_dataset_label_plot.loc[
                    total_durations_positive.index, duration_per_dataset_label_plot.columns
                ]

                # Calculate proportions
                df_proportions = df_proportions_input.apply(
                    lambda x: (x / x.sum() * 100) if x.sum() > 1e-9 else x * 0,
                    axis=1,
                ).fillna(0)

                num_datasets_to_plot = len(total_durations_positive)
                bar_thickness_factor = 0.2  # Adjust for desired bar thickness relative to space
                fig_height = max(0, num_datasets_to_plot * bar_thickness_factor + 2.5)  # +2.5 for titles/margins/legend

                # Swapped axes: ax1 for proportions (left), ax2 for durations (right)
                # Adjusted width_ratios: proportions plot might need more width if many labels
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, fig_height), gridspec_kw={"width_ratios": [3, 2]})

                # --- Panel 1: Label Proportions (Left) ---
                plot_colors_for_proportions = [
                    label_color_map.get(label, default_color) for label in df_proportions.columns
                ]

                if not df_proportions.empty:
                    df_proportions.plot(
                        kind="barh", stacked=True, ax=ax1, color=plot_colors_for_proportions, legend=False, width=0.75
                    )

                # ax1.set_title("Label Proportions", fontsize=14)  # , fontweight='bold')
                ax1.set_xlabel("Segmentation Percentage (%)", fontsize=14)
                ax1.set_xlim(0, 100)
                # ax1.set_ylabel("Dataset", fontsize=12)  # Y-labels on the first (left) plot
                # Enhance tick parameters for modern look
                ax1.tick_params(axis="both", which="major", labelsize=12)
                # Remove top and right spines for a cleaner look (seaborn styles might do this partially)
                ax1.spines["top"].set_visible(False)
                ax1.spines["right"].set_visible(False)
                ax1.grid(False)
                # ax1.grid(
                #    False, axis="x", linestyle="-", alpha=0.7
                # )  # Keep horizontal grid lines for barh (vertical lines on plot)

                # --- Panel 2: Total Duration (Right) ---
                bar_color_duration = "#444444"
                total_durations_positive.plot(kind="barh", ax=ax2, color=bar_color_duration, width=0.75)
                ax2.set_xscale("log")
                # ax2.set_title("Total Duration", fontsize=14)
                ax2.set_xlabel(
                    "Total Single View Duration (log scale)", fontsize=14
                )  # Label changed slightly as ticks define units
                ax2.set_yticklabels([])
                ax2.set_ylabel("")

                # Custom x-axis ticks and labels
                tick_values_seconds = [
                    30 * 60,  # 30 minutes
                    1 * 60 * 60,  # 1 hour
                    2 * 60 * 60,  # 2 hours
                    4 * 60 * 60,  # 4 hours
                    8 * 60 * 60,  # 8 hours
                ]
                tick_labels = ["30 min", "1 h", "2 h", "4 h", "8 h"]

                ax2.set_xticks(tick_values_seconds)
                ax2.set_xticklabels(tick_labels, fontsize=12)

                # Add text annotations for total duration
                for bar in ax2.patches:
                    width = bar.get_width()
                    # Convert seconds to hours and minutes
                    hours = int(width // 3600)
                    minutes = int((width % 3600) // 60)
                    duration_text = ""
                    if hours > 0:
                        duration_text += f"{hours}h "
                    if minutes > 0 or hours == 0:  # Show minutes if > 0 or if hours is 0
                        duration_text += f"{minutes}m"
                    if not duration_text:  # Handle cases with less than 1 minute
                        duration_text = f"{width:.0f}s"

                    # Position the text to the right of the bar
                    # Adjust x_offset_factor based on the log scale if needed
                    # A small constant offset might be simpler to start with
                    x_offset = width * 0.05  # 5% of bar width as offset
                    ax2.text(
                        width + x_offset,  # X position: end of bar + offset
                        bar.get_y() + bar.get_height() / 2,  # Y position: middle of bar
                        duration_text,
                        va="center",
                        ha="left",  # Align text to start after the bar
                        fontsize=12,  # Adjust fontsize as needed
                        color="black",  # Adjust color as needed
                    )

                # Ensure only major grid lines corresponding to the new ticks are shown
                # First, turn off any default grid that might be on from the style
                ax2.grid(False)
                # Then, enable grid for the major ticks on the x-axis (which are now our custom ticks)
                # ax2.grid(True, axis="x", which="major", linestyle="-", alpha=0.4)

                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)

                # --- Legend ---
                handles = []
                labels_for_legend = []
                plotted_labels_in_order = df_proportions.columns.tolist()

                for label in plot_label_order:  # Use the original master label order for legend
                    if label in plotted_labels_in_order:  # Only if the label is actually plotted
                        color = label_color_map.get(label, default_color)
                        handles.append(
                            plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor="none")
                        )  # edgecolor='none' or subtle
                        labels_for_legend.append(label)

                if handles:
                    legend_ncol = 1
                    if len(labels_for_legend) > 8:  # Adjust threshold for multiple columns
                        legend_ncol = 2
                    if len(labels_for_legend) > 16:
                        legend_ncol = 3
                    fig.legend(
                        handles,
                        labels_for_legend,
                        title_fontsize="13",  # , fontweight='bold'),
                        fontsize="11",
                        bbox_to_anchor=(0.99, 0.15),  # Adjusted for potential width changes
                        loc="lower right",
                        ncol=legend_ncol,
                        frameon=False,  # Cleaner legend box
                    )

                # Adjust tight_layout rect: [left, bottom, right, top]
                # Right margin needs to account for the legend if it's outside
                right_margin = 0.88 if handles else 0.97
                fig.tight_layout()

                output_filename = save_dir_path / "dataset_shares_and_durations.pdf"
                plt.savefig(output_filename, dpi=150)  # Slightly higher DPI for better quality
                plt.close(fig)
                print(f"Saved: {output_filename}")
    else:
        print("Skipping Plot 1: No labels defined in plot_label_order or no data to plot.")

    # Plot 2: Relative Label Distribution per Dataset (based on duration)
    if plot_label_order:  # Proceed only if there are labels to plot
        sum_durations_df = (
            df.groupby(["dataset_name", "label_name"])["duration"].sum().reset_index(name="total_label_duration")
        )

        # Total duration per dataset
        total_dataset_duration = (
            sum_durations_df.groupby("dataset_name")["total_label_duration"]
            .sum()
            .reset_index(name="total_duration_in_dataset")
        )

        # Merge to calculate proportion
        label_dist_df = pd.merge(sum_durations_df, total_dataset_duration, on="dataset_name")
        label_dist_df["proportion"] = label_dist_df["total_label_duration"] / label_dist_df["total_duration_in_dataset"]

        # Pivot for stacked bar plot
        pivot_df = label_dist_df.pivot(index="dataset_name", columns="label_name", values="proportion").fillna(0)

        # Ensure all columns from plot_label_order exist, then reorder and select
        for col in plot_label_order:
            if col not in pivot_df:
                pivot_df[col] = 0
        pivot_df = pivot_df.reindex(index=sorted_dataset_names, columns=plot_label_order).fillna(0)

        # Filter out labels that are all zero AFTER reindexing
        pivot_df_plot = pivot_df.loc[:, (pivot_df != 0).any(axis=0)]

        plot2_colors = [label_color_map.get(label, default_color) for label in pivot_df_plot.columns]

        if not pivot_df_plot.empty:
            print("\nDEBUG: Data for Plot 2 (Relative Distribution - Head):")
            print(pivot_df_plot.head())
            # If you want to check specific labels, you can do:
            # if 'Fall' in pivot_df_plot.columns and 'Walk' in pivot_df_plot.columns:
            #     print("\nDEBUG: Specific labels ('Fall', 'Walk') in Plot 2 data:")
            #     print(pivot_df_plot[['Fall', 'Walk']].head())

            fig, ax = plt.subplots(figsize=(14, 10))
            pivot_df_plot.plot(kind="barh", stacked=True, ax=ax, color=plot2_colors)
            ax.set_title("Relative Label Distribution (by Duration) per Dataset")
            ax.set_xlabel("Proportion of Total Duration")
            ax.set_ylabel("Dataset")
            ax.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout
            plt.savefig(save_dir_path / "relative_label_duration_distribution_per_dataset.png")
            plt.close(fig)
            print(f"Saved: {save_dir_path / 'relative_label_duration_distribution_per_dataset.png'}")
        else:
            print("Skipping Plot 2: No data to plot after filtering.")
    else:
        print("Skipping Plot 2: No labels defined in plot_label_order or no data to plot.")

    # Plot 3: Segment duration distribution per label (overall) - Horizontal Violin Plot
    # Filter plot_label_order for violin plot to only include labels actually present in df
    violin_plot_labels_present_in_data = [label for label in plot_label_order if label in actual_label_names_from_data]

    if violin_plot_labels_present_in_data:
        violin_palette = {
            label: label_color_map.get(label, default_color) for label in violin_plot_labels_present_in_data
        }

        plt.figure(figsize=(10, max(6, len(violin_plot_labels_present_in_data) * 0.6)))  # Adjusted height factor
        sns.violinplot(
            data=df,
            y="label_name",
            x="duration",
            palette=violin_palette,
            orient="h",
            order=violin_plot_labels_present_in_data,
            cut=0,  # Prevents violins from extending beyond data range
            inner="quartile",  # Shows quartiles inside violins
        )
        plt.title("Segment Duration Distribution per Label (Overall)")
        plt.ylabel("Label")
        plt.xlabel("Duration (seconds)")
        plt.tight_layout()
        plt.savefig(save_dir_path / "duration_distribution_per_label_violin.png")
        plt.close()
        print(f"Saved: {save_dir_path / 'duration_distribution_per_label_violin.png'}")
    else:
        print("Skipping Plot 3: No relevant labels found in data for violin plot or plot_label_order is empty.")

    print("\nStatistics generation complete.")


if __name__ == "__main__":
    main()
