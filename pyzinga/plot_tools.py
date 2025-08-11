"""Import libraries."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional


"""                     Define settings for plotting.                       """
# Default plotting parameters for plots.
DEFAULT_FIGSIZE = (10, 8)
DEFAULT_HUE = None
DEFAULT_EDGECOLOR = "black"
DEFAULT_LEGEND = False
DEFAULT_STYLE = "darkgrid"
DEFAULT_PALETTE = "deep"
DEFAULT_SAVEDIR = None
DEFAULT_CMAP = "coolwarm"

sns.set_theme(style=DEFAULT_STYLE, palette=DEFAULT_PALETTE)


"""                     Functions.                       """
# Function for count plot.
def count_plot(
    df: pd.DataFrame,
    col: str,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_dir: Optional[str] = DEFAULT_SAVEDIR,
    **kwargs
) -> None:
    """
    Create a count plot for a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name to plot (categorical).
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8).
    save_dir : Optional[str], optional
        Directory to save the plot. If None, plot is not saved.
    **kwargs
        Additional keyword arguments passed to seaborn.countplot.

    Returns
    -------
    None
    """
    # Define plot title and labels.
    title = f"Count (frequency) plot of '{col.replace('_', ' ').title()}'"
    xlabel = f"{col.replace('_', ' ').title()}"
    ylabel = "Count"

    # Fill missing values in the column with 'X' and convert to category type.
    df = df.copy()
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna("X").astype("category")

    # Set defaults if not provided in kwargs
    kwargs.setdefault("palette", DEFAULT_PALETTE)
    kwargs.setdefault("hue", DEFAULT_HUE)
    kwargs.setdefault("edgecolor", DEFAULT_EDGECOLOR)
    kwargs.setdefault("legend", DEFAULT_LEGEND)
    kwargs.setdefault("stat", "count")

    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=df, x=col, ax=ax, **kwargs)
    ax.set_title(label=title)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)

    # Optionally, handle legend for seaborn < 0.12
    # ax.legend().set_visible(False)

    # Save the plot if a directory is specified.
    if save_dir is not None:
        plt.savefig(
            fname=os.path.join(save_dir, f"count_plot_{col.lower()}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Count plot saved at '{os.path.join(save_dir, f'count_plot_{col.lower()}.png')}'"
        )
    plt.show()
    plt.close(fig)

# Function for histogram plot.
def histogram_plot(
    df: pd.DataFrame,
    col: str,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_dir: Optional[str] = DEFAULT_SAVEDIR,
    prefix: Optional[str] = None,
    **kwargs
) -> None:
    """
    Create a histogram plot for a numerical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name to plot (numerical).
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8).
    save_dir : Optional[str], optional
        Directory to save the plot. If None, plot is not saved.
    prefix : Optional[str], optional
        Prefix for the plot title and filename.
    **kwargs
        Additional keyword arguments passed to seaborn.histplot.

    Returns
    -------
    None
    """
    # Define plot title, filename and labels.
    title = f"Histogram Plot of '{col.replace('_', ' ').title()}'"
    filename = f"histogram_plot_{col.replace(' ', '_').lower()}.png"
    if prefix is not None:
        title = f"{prefix} {title}"
        filename = f"{prefix.lower()}_{filename}"
    xlabel = f"{col.replace('_', ' ').title()}"
    ylabel = "Frequency"

    # Set defaults if not provided in kwargs
    kwargs.setdefault("bins", "auto")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("color", "teal")
    kwargs.setdefault("edgecolor", DEFAULT_EDGECOLOR)
    kwargs.setdefault("legend", DEFAULT_LEGEND)

    # Plot the image.
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=df, x=col, ax=ax, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the plot if a directory is specified.
    if save_dir is not None:
        plt.savefig(
            fname=os.path.join(save_dir, filename), dpi=300, bbox_inches="tight"
        )
        print(f"Histogram plot saved at '{os.path.join(save_dir, filename)}'")
    plt.show()
    plt.close(fig)

# Function for box plot.
def box_plot(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    yscale: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_dir: Optional[str] = DEFAULT_SAVEDIR,
    prefix: Optional[str] = None,
    **kwargs
) -> None:
    """
    Create a box plot for one or two columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    x_col : Optional[str], optional
        Column name for x-axis (categorical).
    y_col : Optional[str], optional
        Column name for y-axis (numerical).
    yscale : Optional[str], optional
        Scale for y-axis ('log' for log scale).
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8).
    save_dir : Optional[str], optional
        Directory to save the plot. If None, plot is not saved.
    prefix : Optional[str], optional
        Prefix for the plot title and filename.
    **kwargs
        Additional keyword arguments passed to seaborn.boxplot.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If both x_col and y_col are None.
    """
    # Validate input columns.
    if x_col is None and y_col is None:
        raise ValueError("At least one of 'x_col' or 'y_col' must be provided.")

    # Define plot title and labels.
    if x_col is not None and y_col is not None:
        title = f"Box Plot of '{y_col.replace('_', ' ').title()}' w.r.t. '{x_col.replace('_', ' ').title()}'"
        if prefix is not None:
            title = f"{prefix} {title}"

        xlabel = f"{x_col.replace('_', ' ').title()}"
        ylabel = (
            f"{y_col.replace('_', ' ').title()}"
            if yscale is None
            else f"{y_col.replace('_', ' ').title()} (log scale)"
        )

        filename = f"box_plot_{x_col.lower()}_vs_{y_col.lower()}.png"
    elif x_col is not None:
        title = f"Box Plot of '{x_col.replace('_', ' ').title()}'"
        if prefix is not None:
            title = f"{prefix} {title}"

        xlabel = f"{x_col.replace('_', ' ').title()}"
        ylabel = None
        filename = f"box_plot_{x_col.lower()}.png"
    else:
        title = f"Box Plot of '{y_col.replace('_', ' ').title()}'"
        if prefix is not None:
            title = f"{prefix} {title}"

        xlabel = None
        ylabel = (
            f"{y_col.replace('_', ' ').title()}"
            if yscale is None
            else f"{y_col.replace('_', ' ').title()} (log scale)"
        )
        filename = f"box_plot_{y_col.lower()}.png"
    if prefix is not None:
        filename = f"{prefix.lower()}_{filename}"

    # Normalize filename
    filename = filename.replace(" ", "_")

    # Set defaults if not provided in kwargs
    kwargs.setdefault("palette", "Set2")
    kwargs.setdefault("hue", x_col)

    # Plot the image.
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, **kwargs)
    if yscale == "log":
        plt.yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the plot if a directory is specified.
    if save_dir is not None:
        plt.savefig(
            fname=os.path.join(save_dir, filename), dpi=300, bbox_inches="tight"
        )
        print(f"Box plot saved at '{os.path.join(save_dir, filename)}'")
    plt.show()
    plt.close(fig)

# Function for correlation heatmap.
def corr_heatmap_plot(
    df: pd.DataFrame,
    save_dir: Optional[str] = DEFAULT_SAVEDIR,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    mask_upper: bool = True,
    **kwargs
) -> None:
    """
    Create a correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    save_dir : Optional[str], optional
        Directory to save the plot. If None, plot is not saved.
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8).
    mask_upper : bool, optional
        If True, mask the upper triangle of the heatmap.
    **kwargs
        Additional keyword arguments passed to seaborn.heatmap.

    Returns
    -------
    None
    """
    # Define plot title.
    title = "Correlation Heatmap (masked)" if mask_upper else "Correlation Heatmap"
    filename = (
        "correlation_heatmap_masked.png" if mask_upper else "correlation_heatmap.png"
    )

    # Ensure the DataFrame contains only numeric columns for correlation.
    df = df.select_dtypes(include=["number"])
    corr = df.corr()  # Correlation matrix.
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

    # Set defaults if not provided in kwargs
    kwargs.setdefault("cmap", DEFAULT_CMAP)
    kwargs.setdefault("cbar_kws", {"shrink": 1.0})

    # Plot the image.
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data=corr, annot=True, fmt=".2f", ax=ax, square=True, mask=mask, **kwargs
    )
    ax.set_title(title)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    ax.set_aspect("equal")  # Ensure each cell is square and grid aligns
    ax.grid(True, which="both", axis="both")  # Explicitly enable grid lines

    # Save the plot if a directory is specified.
    if save_dir is not None:
        os.makedirs(name=save_dir, exist_ok=True)
        plt.savefig(
            fname=os.path.join(save_dir, filename), dpi=300, bbox_inches="tight"
        )
        print(f"Correlation heatmap saved at '{os.path.join(save_dir, filename)}'")

    plt.show()
    plt.close(fig)
