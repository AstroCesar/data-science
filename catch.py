import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
##############################    INPUT USER SECTION     ##################################
##########################################################################################

name_file   = 'input.txt'        # input file (text/ascii format)
name_output = 'output.txt'       # output file name

colx = 1   # column index for X  (RA, pmRA, or X)   — 1-based
coly = 2   # column index for Y  (Dec, pmDec, or Y)  — 1-based

# ---- MODE -------------------------------------------------------------------
# option = 1  →  spherical circle  : sky coordinates  (RA / Dec in degrees)
# option = 2  →  flat circle       : linear/proper-motion plane
# option = 3  →  flat square       : linear/proper-motion plane
option = 1

# ---- CENTER -----------------------------------------------------------------
centerX_user = 260.994124    # RA  (deg) | pmRA  (mas/yr) | X
centerY_user = -26.353417    # Dec (deg) | pmDec (mas/yr) | Y

# ---- CAPTURE REGION ---------------------------------------------------------
radio_user = 0.5    # radius  — degrees (option 1) or same units as data (option 2)
lado_user  = 1.0    # side    — same units as data (option 3 only)

# ---- PLOT -------------------------------------------------------------------
plot_show = True    # True → show plot after running

##########################################################################################
##########################################################################################
##########################################################################################

df    = np.genfromtxt(name_file)   # load data
x     = df[:, colx - 1]
y     = df[:, coly - 1]
l_col = df.shape[1]


# =============================================================================
#  FUNCTIONS
# =============================================================================

def circulo_esferico(center_ra, center_dec, radio_deg, arreglo, col_ra=1, col_dec=2):
    """
    Captures rows inside a spherical circle on the sky.

    Parameters
    ----------
    center_ra  : float  — RA of the centre (degrees)
    center_dec : float  — Dec of the centre (degrees)
    radio_deg  : float  — search radius (degrees)
    arreglo    : ndarray — full data array
    col_ra     : int    — 1-based column index for RA
    col_dec    : int    — 1-based column index for Dec

    Returns
    -------
    output     : ndarray — rows inside the circle
    ras_out    : ndarray — RA  values of captured rows
    decs_out   : ndarray — Dec values of captured rows
    indices    : ndarray — row indices of captured rows
    """
    ras  = arreglo[:, col_ra  - 1]
    decs = arreglo[:, col_dec - 1]

    center_ra_rad  = np.radians(center_ra)
    center_dec_rad = np.radians(center_dec)
    ras_rad        = np.radians(ras)
    decs_rad       = np.radians(decs)

    # Spherical law of cosines (safe for small fields when using float64)
    cos_ang = (np.sin(center_dec_rad) * np.sin(decs_rad) +
               np.cos(center_dec_rad) * np.cos(decs_rad) *
               np.cos(ras_rad - center_ra_rad))
    cos_ang    = np.clip(cos_ang, -1.0, 1.0)   # numerical safety
    ang_dist   = np.degrees(np.arccos(cos_ang))

    indices    = np.where(ang_dist < radio_deg)[0]
    output     = arreglo[indices]
    return output, output[:, col_ra - 1], output[:, col_dec - 1], indices


def circulo_plano(center_x, center_y, radio, arreglo, col_x=1, col_y=2):
    """
    Captures rows inside a flat (Euclidean) circle.

    Parameters
    ----------
    center_x / center_y : float  — centre coordinates
    radio               : float  — radius (same units as data)
    arreglo             : ndarray — full data array
    col_x / col_y       : int    — 1-based column indices

    Returns
    -------
    output   : ndarray — rows inside the circle
    xs_out   : ndarray — X values of captured rows
    ys_out   : ndarray — Y values of captured rows
    indices  : ndarray — row indices of captured rows
    """
    xs = arreglo[:, col_x - 1]
    ys = arreglo[:, col_y - 1]

    dist2   = (xs - center_x)**2 + (ys - center_y)**2
    indices = np.where(dist2 < radio**2)[0]
    output  = arreglo[indices]
    return output, output[:, col_x - 1], output[:, col_y - 1], indices


def cuadrado(center_x, center_y, lado, arreglo, col_x=1, col_y=2):
    """
    Captures rows inside a flat square.

    Parameters
    ----------
    center_x / center_y : float  — centre coordinates
    lado                : float  — side length (same units as data)
    arreglo             : ndarray — full data array
    col_x / col_y       : int    — 1-based column indices

    Returns
    -------
    output   : ndarray — rows inside the square
    xs_out   : ndarray — X values of captured rows
    ys_out   : ndarray — Y values of captured rows
    indices  : ndarray — row indices of captured rows
    """
    xs = arreglo[:, col_x - 1]
    ys = arreglo[:, col_y - 1]

    half    = lado / 2.0
    mask_x  = np.abs(xs - center_x) < half
    mask_y  = np.abs(ys - center_y) < half
    indices = np.where(mask_x & mask_y)[0]
    output  = arreglo[indices]
    return output, output[:, col_x - 1], output[:, col_y - 1], indices


# =============================================================================
#  MAIN — run mode selected by user
# =============================================================================

if option == 1:
    opt    = "sky_circ_"
    label  = f"Spherical circle  |  centre ({centerX_user}, {centerY_user})  r = {radio_user}°"
    output_arr, cap_x, cap_y, idx = circulo_esferico(
        centerX_user, centerY_user, radio_user, df, colx, coly)
    margin = radio_user * 1.7

elif option == 2:
    opt    = "flat_circ_"
    label  = f"Flat circle  |  centre ({centerX_user}, {centerY_user})  r = {radio_user}"
    output_arr, cap_x, cap_y, idx = circulo_plano(
        centerX_user, centerY_user, radio_user, df, colx, coly)
    margin = radio_user * 1.7

elif option == 3:
    opt    = "flat_squ_"
    label  = f"Flat square  |  centre ({centerX_user}, {centerY_user})  side = {lado_user}"
    output_arr, cap_x, cap_y, idx = cuadrado(
        centerX_user, centerY_user, lado_user, df, colx, coly)
    margin = lado_user * 0.7

else:
    raise ValueError("option must be 1, 2, or 3")

print(f"[catch]  mode: {opt}  |  {len(idx)} objects captured")

# =============================================================================
#  OUTPUT FILE
# =============================================================================

fmt = "%11.7f %11.7f" + " %10.5f" * (l_col - 2)
np.savetxt(opt + name_output, output_arr, fmt=fmt,
           header=label, comments="# ")
print(f"[catch]  saved → {opt + name_output}")

# =============================================================================
#  PLOT
# =============================================================================

if plot_show:
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(x, y, s=5, color="steelblue", alpha=0.5, label="all data")
    ax.scatter(cap_x, cap_y, s=15, color="tomato", zorder=3, label=f"captured ({len(idx)})")
    ax.scatter(centerX_user, centerY_user, marker="+", s=120,
               color="black", zorder=4, label="centre")

    if option in (1, 2):
        # draw circle outline
        theta  = np.linspace(0, 2 * np.pi, 300)
        r      = radio_user
        circ_x = centerX_user + r * np.cos(theta)
        circ_y = centerY_user + r * np.sin(theta)
        ax.plot(circ_x, circ_y, "k--", lw=0.9, alpha=0.7)
    else:
        # draw square outline
        half = lado_user / 2.0
        sq_x = [centerX_user - half, centerX_user + half,
                centerX_user + half, centerX_user - half, centerX_user - half]
        sq_y = [centerY_user - half, centerY_user - half,
                centerY_user + half, centerY_user + half, centerY_user - half]
        ax.plot(sq_x, sq_y, "k--", lw=0.9, alpha=0.7)

    ax.set_xlim(centerX_user - margin, centerX_user + margin)
    ax.set_ylim(centerY_user - margin, centerY_user + margin)

    xlabel = "RA (deg)" if option == 1 else "X"
    ylabel = "Dec (deg)" if option == 1 else "Y"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=9)
    ax.invert_xaxis()   # astronomical convention (RA increases to the left)
    plt.tight_layout()
    plt.savefig(opt + name_output.replace(".txt", "") + "_plot.png", dpi=150)
    plt.show()
    print(f"[catch]  plot  → {opt + name_output.replace('.txt','')}_plot.png")
