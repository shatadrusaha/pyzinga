"""                     Import libraries.                       """
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from typing import Tuple, Optional  # noqa: E402
import os  # noqa: E402

# Function for plotting correlation matrix.
def plot_correlation_matrix(
        df: pd.DataFrame, 
        figsize: Tuple[float, float] = (10, 8),
        col_target: Optional[str] = None,
        folder_tosave_plot: Optional[str] = None,
        cmap: str = 'coolwarm',
        annot_fmt: str = ".2f",
        show_both_masks: bool = True
    ) -> pd.DataFrame:
    """
    Plot correlation matrix or correlation with respect to a target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the features to calculate correlation for.
    figsize : tuple of (float, float), default=(10, 8)
        Figure size in inches as a tuple of (width, height).
    col_target : str, optional
        If provided, plot correlation of all features with respect to this target column.
        Must be a numeric column in the DataFrame.
    folder_tosave_plot : str, optional
        If provided, save the figure to this folder. The folder will be created if it doesn't exist.
    cmap : str, default='coolwarm'
        Colormap name to use for the heatmap.
    annot_fmt : str, default='.2f'
        String formatting code for the annotations.
    show_both_masks : bool, default=True
        If True, show both masked and unmasked correlation matrices when col_target is None.
        If False, only show the unmasked version.
    
    Returns
    -------
    pd.DataFrame
        The correlation DataFrame.
    
    Raises
    ------
    ValueError
        If the DataFrame is empty, target column is not found, or target column
        is not numeric.
    TypeError
        If inputs are not of the correct types.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': np.random.rand(100),
    ...     'B': np.random.rand(100),
    ...     'C': np.random.rand(100)
    ... })
    >>> corr = plot_correlation_matrix(df)
    >>> corr = plot_correlation_matrix(df, col_target='A')
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
        
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot compute correlation matrix.")
        
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("figsize must be a tuple of two numbers")
        
    if not all(isinstance(x, (int, float)) and x > 0 for x in figsize):
        raise ValueError("figsize values must be positive numbers")
        
    if folder_tosave_plot is not None and not isinstance(folder_tosave_plot, str):
        raise TypeError("folder_tosave_plot must be None or a string")
        
    if not isinstance(cmap, str):
        raise TypeError("cmap must be a string")

    # Set the default font size based on the figure size.
    fontsize = figsize[0] * 1.5

    # Create folder if it doesn't exist
    if folder_tosave_plot is not None:
        if not os.path.exists(folder_tosave_plot):
            os.makedirs(folder_tosave_plot)
            print(f"Created directory '{folder_tosave_plot}' for saving plots.")

    if col_target is not None:
        if col_target not in df.columns:
            raise ValueError(f"Target column '{col_target}' not found in the DataFrame.")
        if df[col_target].dtype not in [np.float64, np.int64]:
            raise ValueError(f"Target column '{col_target}' must be numeric (float or int).")
        
        # Calculate the correlation with respect to the target column.
        corr = df.corr()[col_target].drop(labels=col_target, axis=0).sort_values(ascending=False)

        # Plot correlation with respect to the target column.
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=corr.values, y=corr.index, hue=corr.index, palette=cmap, legend=False, ax=ax)
        ax.set_title(f"Correlation Plot w.r.t. Target Column ('{col_target}')", fontsize=fontsize, loc='center')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Features')
        plt.tight_layout()
        
        if folder_tosave_plot is not None:
            filename_tosave = "correlation_plot_wrt_target.png"
            plt.savefig(os.path.join(folder_tosave_plot, filename_tosave), 
                        dpi=600, 
                        bbox_inches='tight')
        
        plt.show()
        plt.close(fig)

    else:
        # Calculate the correlation matrix for the entire DataFrame.
        corr = df.corr()

        # Mask the upper triangle of the correlation matrix.
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Plot and save (if applicable) the correlation matrix.
        plot_types = [mask, None] if show_both_masks else [None]
        
        for plot_type in plot_types:
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                data=corr, 
                annot=True, 
                cmap=cmap, 
                fmt=annot_fmt, 
                mask=plot_type,
                ax=ax
            )
            
            title = "Correlation Matrix"
            if plot_type is not None:
                title += " (with mask)"
                filename_tosave = "correlation_matrix_masked.png"
            else:
                filename_tosave = "correlation_matrix.png"
            ax.set_title(title, fontsize=fontsize, loc='center')
            
            plt.tight_layout()
            
            if folder_tosave_plot is not None:
                plt.savefig(os.path.join(folder_tosave_plot, filename_tosave), 
                            dpi=600, # Save as PNG with high resolution.
                            bbox_inches='tight') 
            
            plt.show()
            plt.close(fig)
    
    return corr