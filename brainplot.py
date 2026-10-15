# %%
from surfplot import Plot
from brainspace.datasets import load_parcellation
from neuromaps.datasets import fetch_fslr
import numpy as np

# %%
def plot_schaefer_fsLR(
        data,
        vmin,
        vmax,
        hemi=None,
        surf_type='veryinflated',
        cmap='plasma',
        resolution=400):

    """
    Plot Schaefer parcellated data onto fsLR surface.

    Parameters
    ----------
    data : (n_parcel,)
        parcel values (e.g., 400×1 vector)
    vmin, vmax : int/float
        color range
    hemi : 'left' | 'right' | 'both'
    surf_type : fsLR surface type
    cmap : matplotlib colormap
    resolution : Schaefer resolution
    """

    # -----------------------------
    # Load fsLR surfaces
    # -----------------------------
    surfaces = fetch_fslr()
    lh, rh = surfaces[surf_type]

    # -----------------------------
    # Load Schaefer atlas
    # -----------------------------
    atlas_hemi = load_parcellation('schaefer', resolution, join=False)
    atlas_both = load_parcellation('schaefer', resolution, join=True)

    if hemi == 'left':
        atlas = atlas_hemi[0]
    elif hemi == 'right':
        atlas = atlas_hemi[1]
    elif hemi == 'both':
        atlas = atlas_both
    else:
        raise ValueError("hemi must be 'left', 'right', or 'both'")

    # -----------------------------
    # Map parcel values → vertices
    # -----------------------------
    unique = np.unique(atlas)
    unique = unique[unique != 0]   # remove medial wall

    plot_data = np.zeros_like(atlas, dtype=float)

    for i, parcel_id in enumerate(unique):
        plot_data[atlas == parcel_id] = data[i]

    # -----------------------------
    # Create plot
    # -----------------------------
    if hemi == 'left':
        p = Plot(
            lh,
            views=['lateral', 'medial'],
            zoom=1.2,
            size=(1200, 800),
            brightness=0.7
        )

    elif hemi == 'right':
        p = Plot(
            rh,
            views=['lateral', 'medial'],
            zoom=1.2,
            size=(1200, 800),
            brightness=0.7
        )

    elif hemi == 'both':
        p = Plot(
            lh,
            rh,
            layout='grid',
            views=['lateral', 'medial'],
            zoom=1.2,
            size=(1600, 800),
            brightness=0.7
        )

    # ---- IMPORTANT: discrete color alignment ----
 # --- main colored layer ---
    p.add_layer(
        plot_data,
        cmap=cmap,
        color_range=(vmin, vmax),
        cbar=True
    )

    # --- parcel outline layer (ADD THIS) ---
    # p.add_layer(
    #     atlas,
    #     cmap='gray',
    #     as_outline=True,
    #     cbar=False
    # )
    fig = p.build()

    return fig


# %%
# example

# fig = plot_schaefer_fsLR(
#     data=data,
#     vmin=0,
#     vmax=vmax,
#     hemi='both',
#     cmap=cmap,
#     resolution=400
# )

