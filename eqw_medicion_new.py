"""
eqw_medicion_corregido.py
=========================
Script para medir anchos equivalentes (EQW) sobre espectros estelares en formato FITS.
Ajusta gaussianas a líneas de absorción y guarda los resultados en formato MOOG.

Traducido originalmente de SuperMongo. Versión corregida con comentarios detallados.
"""

from PyAstronomy import pyasl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def yes_no(question_to_be_answered):
    """Pide al usuario una respuesta sí/no. Retorna True o False."""
    while True:
        choice = input(question_to_be_answered).lower()
        if choice[:1] == 'y':
            return True
        elif choice[:1] == 'n':
            return False
        else:
            print("Please respond with 'Yes' or 'No'\n")


def rangee(lim_i, lim_s, default, question_to_be_ans):
    """
    Pide al usuario un número dentro de un rango [lim_i, lim_s].
    Si el usuario presiona Enter sin escribir nada, usa el valor 'default'.
    Retorna una tupla (False, valor).
    """
    while True:
        user_number = input(question_to_be_ans)
        if not user_number:
            user_number = default
        try:
            val = float(user_number)
            if lim_i <= val <= lim_s:
                return False, val
            else:
                print("Out of range")
        except ValueError:
            print("It's not a number, try again.")


def gauss_mod(c, lline, fline, xgaussmin, ygaussmin, sigline):
    """
    Ajuste gaussiano por mínimos cuadrados no lineales (método iterativo).

    FIX 1: Las variables lline, fline, xgaussmin, ygaussmin y sigline
    antes eran globales implícitas — ahora se pasan como parámetros explícitos.
    Esto evita bugs difíciles de rastrear si el estado global cambia entre llamadas.

    Parámetros:
        c         : nivel del continuo
        lline     : array de longitudes de onda de la región de captura
        fline     : array de flujos correspondientes
        xgaussmin : longitud de onda del mínimo (centro inicial de la gaussiana)
        ygaussmin : flujo en el mínimo
        sigline   : sigma inicial estimado desde la resolución del instrumento

    Retorna:
        A  : amplitud de la gaussiana (profundidad respecto al continuo)
        xo : centro ajustado
        si : sigma ajustado
    """
    n_pts = np.size(lline)

    alfa   = np.zeros(n_pts)
    beta   = np.zeros(n_pts)
    gammap = np.zeros(n_pts)
    delta  = np.zeros(n_pts)
    me1m   = np.zeros(n_pts)

    # FIX 2: me1 y scme1 se inicializaban en 100, lo que hacía que la condición
    # de parada "scme1 > 0.001" fuera siempre verdadera al comienzo incluso si
    # el ajuste ya era bueno. np.inf es el valor correcto para "todavía no calculado".
    me1   = np.inf
    scme1 = np.inf

    A  = c - ygaussmin   # amplitud inicial: diferencia continuo - mínimo
    xo = xgaussmin       # centro inicial: posición del mínimo
    si = sigline         # sigma inicial: estimado del instrumento

    contr = 1  # contador de iteraciones (máximo 20)

    # FIX 3: Se reemplazó "&" (bitwise) por "and" (lógico) para condiciones escalares.
    # Con "&" funciona accidentalmente, pero es semánticamente incorrecto en Python.
    while (scme1 > 0.001) and (contr < 20):

        # Derivadas parciales de la función gaussiana respecto a A, xo, si
        for i in range(n_pts - 1):
            alfa[i]   = fline[i] - (c - A * np.exp(-(lline[i] - xo)**2 / (2 * si**2)))
            beta[i]   = -(np.exp(-(lline[i] - xo)**2 / (2 * si**2)))
            gammap[i] = -(A / si**2) * (lline[i] - xo) * np.exp(-(lline[i] - xo)**2 / (2 * si**2))
            delta[i]  = -((A * (lline[i] - xo)**2) / (si**3)) * np.exp(-(lline[i] - xo)**2 / (2 * si**2))

        # Construcción de la matriz normal y el vector independiente
        M = np.zeros((3, 3))
        V = np.zeros(3)

        M[0, 0] = np.sum(beta**2)
        M[0, 1] = np.sum(beta * gammap)
        M[0, 2] = np.sum(beta * delta)
        M[1, 0] = np.sum(beta * gammap)
        M[1, 1] = np.sum(gammap**2)
        M[1, 2] = np.sum(gammap * delta)
        M[2, 0] = np.sum(beta * delta)
        M[2, 1] = np.sum(gammap * delta)
        M[2, 2] = np.sum(delta**2)

        V[0] = np.sum(beta * alfa)
        V[1] = np.sum(gammap * alfa)
        V[2] = np.sum(delta * alfa)

        # Resolución del sistema por pseudoinversa (robusto ante matrices singulares)
        Minv = np.linalg.pinv(M)
        C = Minv.dot(V)

        # Actualización de parámetros
        A  += C[0]
        xo += C[1]
        si += C[2]

        # Cálculo del residuo cuadrático medio para evaluar convergencia
        for j in range(n_pts - 1):
            me1m[j] = fline[j] - (c - A * np.exp(-(lline[j] - xo)**2 / (2 * si**2)))

        me1_park = me1
        me1      = np.sqrt(1 / ((n_pts**2) - 5) * np.sum(me1m**2))
        scme1    = np.abs(me1_park - me1)
        contr   += 1

    return A, xo, si


# =============================================================================
# PARÁMETROS DE ENTRADA
# =============================================================================

file_name_spectra_fit = input('Enter the FITS filename (e.g. my_spectrum.fits): ').strip()
filename_line_list_Fe     = 'linelist_fe.dat'        # lista de líneas de Fe
filename_line_list_other  = 'linelist_other.dat'     # lista de líneas de otros elementos

# Nombre base para los archivos de salida (sin extensión .fits)
name_out = file_name_spectra_fit.split('.fits')[0]

# Parámetros del instrumento
Rs   = 0.25   # resolución espectral (FWHM en Angstroms, aprox.)
lint = 3      # semiventana de visualización en el plot (en Å)
lyi  = 0.0    # límite inferior del eje Y (flujo)
lys  = 1.55   # límite superior del eje Y (flujo)

# Lectura del espectro FITS
l, f = pyasl.read1dFitsSpec(file_name_spectra_fit)
minw = l[0]    # longitud de onda mínima del espectro
maxw = l[-1]   # longitud de onda máxima del espectro


# =============================================================================
# LECTURA DE LA LISTA DE LÍNEAS
# Formato esperado: lambda | numb_elemento | ep | loggf | damp
# (header en la fila 1, datos desde la fila 2)
# =============================================================================

cond = True
while cond:
    choice_list = input('What line-list want to measure: Fe (1) or others (2)? ').lower()

    if choice_list == '1':
        # FIX 4: delim_whitespace=True está deprecado en pandas >= 2.0.
        # Se reemplaza por sep=r'\s+' que es equivalente y compatible con versiones modernas.
        linelistFE = pd.read_table(
            filename_line_list_Fe,
            sep=r'\s+',
            names=['lambda', 'el', 'ep', 'loggf', 'damp'],
            header=0   # la fila 0 es el header (se saltea)
        )
        linelist_array = np.array(linelistFE)
        el_name = 'Fe'
        cond = False

    elif choice_list == '2':
        linelist_other = pd.read_table(
            filename_line_list_other,
            sep=r'\s+',
            names=['lambda', 'el', 'ep', 'loggf', 'damp'],
            header=0
        )
        linelist_array = np.array(linelist_other)
        el_name = 'Other'
        cond = False

    else:
        print("Invalid option, enter 1 or 2.")


# =============================================================================
# FILTRADO: solo líneas dentro del rango del espectro
# =============================================================================

indice_range = np.where((linelist_array[:, 0] > minw) & (linelist_array[:, 0] < maxw))

lambda_new = linelist_array[indice_range, 0][0]
el_new     = linelist_array[indice_range, 1][0]
ep_new     = linelist_array[indice_range, 2][0]
loggf_new  = linelist_array[indice_range, 3][0]
damp_new   = linelist_array[indice_range, 4][0]


# =============================================================================
# INICIALIZACIÓN DE ARRAYS Y CONTADORES
# =============================================================================

c            = 1      # nivel del continuo (valor inicial)
infoeqw      = np.zeros(np.size(lambda_new))   # EQW medidos (en Å)
si_back      = np.zeros(np.size(lambda_new))   # sigma ajustado de cada línea
cont_default = 1      # (no se usa actualmente, reservado)
rng          = 0.5    # rango de captura inicial (en Å, a cada lado del centro)
k            = 0      # índice de línea actual en la lista
sav          = 0      # contador de líneas guardadas


# =============================================================================
# LOOP PRINCIPAL: medición interactiva línea por línea
# =============================================================================

# FIX 5: La condición original era "k != (np.size(lambda_new) - 1)", lo que hacía
# que el loop terminara ANTES de medir la última línea de la lista.
# La condición correcta es "k < np.size(lambda_new)".
while k < np.size(lambda_new):

    # El usuario puede ajustar el rango de captura y el continuo en cada línea
    boole, rng = rangee(0.3, 4, rng,
                        'Enter the range to capture (0.3 - 4), prev.: ' + str(rng) + ', press enter to keep: ')
    boole2, c  = rangee(-2, 2, c,
                        'Enter the continuum, prev.: ' + str(c) + ', press enter to keep: ')

    # --- Extracción de la región de captura ---
    # FIX 6: En el código original, lline_ind, lline y fline se calculaban DOS veces
    # con el mismo código (líneas redundantes). Se eliminó el primer bloque duplicado.
    lline_ind = np.where((l > (lambda_new[k] - rng / 2)) & (l < (lambda_new[k] + rng / 2)))
    lline = l[lline_ind]
    fline = f[lline_ind]

    # Mínimo observado dentro del rango de captura (estimación inicial del centro)
    xgaussmin = lline[np.argmin(fline)]
    ygaussmin = fline[np.argmin(fline)]

    # Sigma inicial estimado a partir de la resolución del instrumento (FWHM → sigma)
    sigline = Rs / (2 * 1.1774)

    # Ventana de visualización (más amplia que la captura)
    lxi = lambda_new[k] - lint   # límite izquierdo del plot
    lxs = lambda_new[k] + lint   # límite derecho del plot

    # Datos dentro de la ventana de visualización
    indice_range_2 = np.where((l > lxi) & (l < lxs))
    ll = l[indice_range_2]
    fl = f[indice_range_2]

    # Línea vertical de referencia en la posición nominal de la línea espectral
    lineavert_y = np.arange(lyi, lys + 2, 1)
    lineavert_x = np.full(np.size(lineavert_y), lambda_new[k])

    # --- Ajuste gaussiano ---
    # FIX 1 aplicado: se pasan todos los parámetros explícitamente
    A, xo, si = gauss_mod(c, lline, fline, xgaussmin, ygaussmin, sigline)

    # Gaussiana ajustada evaluada en grilla fina para el plot
    xgauss = np.arange(lxi, lxs, 0.001)
    ygauss = c - A * np.exp(-(xgauss - xo)**2 / (2 * si**2))

    # --- Cálculo del EQW ---
    # Nota: se integra sobre la gaussiana ajustada con paso fijo de 0.001 Å.
    # Esto es una aproximación; para mayor rigor se podría usar np.trapz sobre lline/fline.
    infoeqw[k] = np.sum(((c - ygauss) / c) * 0.001)
    si_back[k] = si

    # --- Plot ---
    # FIX 7: plt.ion() estaba dentro del loop, lo que puede causar problemas en
    # algunos backends. Se mantiene pero se usa show(block=False) + pause para
    # asegurarse de que el plot aparezca antes del input.
    plt.figure(figsize=(18, 6), dpi=70)

    # Panel izquierdo: vista amplia del espectro
    plt.subplot2grid((1, 2), (0, 0))
    plt.plot(l, f, color='blue')
    plt.plot(lineavert_x, lineavert_y, c='red')
    plt.plot(xgauss, ygauss, c='orange')
    plt.xlim(lxi, lxs)
    plt.ylim(lyi, lys)
    plt.xlabel('$\\lambda [\\AA$]', fontsize=13)
    plt.ylabel('Flux', fontsize=13)

    text     = 'Element: ' + str(el_new[k]) + '   W: ' + str(lambda_new[k]) + '[$\\AA$]'
    text2    = 'Contin.: ' + str(c)
    text4    = str(k + 1) + '/' + str(np.size(lambda_new))   # FIX: +1 era incorrecto (off by one en el total)
    textsav  = 'Lines saved: ' + str(sav)

    plt.text(lxi, lys + 0.1, text, fontsize=14)
    plt.text(lxs, lys + 0.1, text2, fontsize=14, horizontalalignment='right')
    plt.text(lxi, lyi - 0.15, text4, fontsize=14)
    plt.text(lxs, lyi - 0.15, textsav, fontsize=14, horizontalalignment='right')
    plt.yticks(np.arange(lyi, lys, 0.1))

    # Panel derecho: zoom en la región de captura
    plt.subplot2grid((1, 2), (0, 1))
    text3 = 'Range: ' + str(rng)
    plt.text(lxi + lint - rng, lys + 0.1, text3, fontsize=14)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.title('Range Capture')
    plt.scatter(l, f, color='blue')
    plt.scatter(lline, fline, c='red')
    plt.plot(lineavert_x, lineavert_y, c='red')
    plt.plot(xgauss, ygauss, c='orange')
    plt.xlim(lxi + lint - rng, lxs - lint + rng)
    plt.ylim(lyi, lys)
    plt.xlabel('$\\lambda [\\AA$]', fontsize=13)
    plt.yticks(np.arange(lyi, lys, 0.1))

    plt.show(block=False)
    plt.pause(0.1)   # FIX 7: pausa necesaria para que el plot aparezca antes del input()

    # --- Decisión del usuario ---
    loop = yes_no('Do you want to repeat the process for this line? (yes/no): ')

    if not loop:
        safe = yes_no('Do you want to save this line? (yes/no): ')
        if safe:
            sav += 1
        else:
            infoeqw[k] = 9999   # valor centinela: esta línea NO se guardará
        k += 1   # avanzar a la siguiente línea solo cuando el usuario confirma

    plt.close()


# =============================================================================
# GUARDADO: mediciones válidas en formato .eqw.dat
# Criterio de validez: 0.00001 < EQW < 2 Å, sigma < 100
# =============================================================================

index_val   = np.where((infoeqw < 2) & (infoeqw > 0.00001) & (si_back < 100))
infoeqwNew  = infoeqw * 1000   # conversión de Å a mÅ

formato = "%4.3f %2.1f %2.3f %2.3f  %2.3f %1.1f %3.1f"
np.savetxt(
    name_out + '.' + el_name + '.eqw.dat',
    np.c_[lambda_new[index_val], el_new[index_val], ep_new[index_val],
          loggf_new[index_val],  damp_new[index_val],
          infoeqwNew[index_val], si_back[index_val]],
    fmt=formato
)

# Guardado de arrays filtrados para el paso siguiente (sigma rejection)
lambda_new_s  = lambda_new[index_val]
ep_new_s      = ep_new[index_val]
el_new_s      = el_new[index_val]
loggf_new_s   = loggf_new[index_val]
damp_new_s    = damp_new[index_val]
infoeqwNew_s  = infoeqwNew[index_val]
si_back_s     = si_back[index_val]


# =============================================================================
# PASO OPCIONAL: SIGMA REJECTION
# Ajusta un polinomio de grado 2 a sigma vs lambda y rechaza outliers.
# Útil para eliminar líneas con ajustes gaussianos pobres o contaminadas.
# =============================================================================

exit_q = yes_no('Do you want to continue to Sigma Rejection? (y/n): ')
if not exit_q:
    sys.exit()

loop_sigma = True
while loop_sigma:

    n = 2   # grado del polinomio de ajuste
    p, C_p = np.polyfit(lambda_new_s, si_back_s, n, cov=True)

    t  = np.array(lambda_new_s)
    ss = si_back_s

    # Evaluación del polinomio e incertezas
    TT    = np.vstack([t**(n - i) for i in range(n + 1)]).T
    yi    = np.dot(TT, p)
    C_yi  = np.dot(TT, np.dot(C_p, TT.T))
    sig_yi = np.sqrt(np.diag(C_yi))

    boole3, nsig = rangee(0, 1000, 1, 'Sigma rejection threshold (positive number): ')

    # Líneas dentro del umbral nsig*sigma
    poly_m   = sum(p[i] * t**(n - i) for i in range(n + 1))
    diffsip  = ss - poly_m
    indicesig = np.where(np.abs(diffsip) < nsig * sig_yi)

    # Plot del ajuste
    plt.figure(figsize=(9, 6), dpi=100)
    plt.title("Polynomial fit (deg {}) ± {} σ".format(n, nsig))
    plt.fill_between(t, yi + nsig * sig_yi, yi - nsig * sig_yi, alpha=0.25)
    plt.plot(t, yi, '-')

    plt.scatter(t, ss, marker='x', c='orange', label='all lines')
    plt.scatter(t[indicesig], ss[indicesig], c='green', marker='x', label='kept lines')
    plt.legend()

    plt.xlabel('$\\lambda [\\AA$]', fontsize=13)
    plt.ylabel('$\\sigma$', fontsize=13)

    # FIX 8: El xlim estaba hardcodeado en 5000–5300 Å, lo que dejaba el plot vacío
    # para espectros fuera de esa región. Ahora usa el rango real de las líneas medidas.
    plt.xlim(np.min(t[indicesig]) - 5, np.max(t[indicesig]) + 5)
    plt.ylim(np.min(ss[indicesig]) - 0.03, np.max(ss[indicesig]) + 0.03)

    plt.show(block=False)
    plt.pause(0.1)

    loop_sigma = yes_no('Do you want to change the sigma threshold? (yes/no): ')
    plt.close()


# =============================================================================
# GUARDADO FINAL: formato MOOG (ordenado por elemento)
# =============================================================================

ep_new2      = ep_new_s[indicesig]
loggf_new2   = loggf_new_s[indicesig]
damp_new2    = damp_new_s[indicesig]
infoeqwNew2  = infoeqwNew_s[indicesig]
el_new2      = el_new_s[indicesig]
lambnew2     = lambda_new_s[indicesig]
si_back2     = si_back_s[indicesig]
disp2        = si_back2 * 0   # columna de dispersión (reservada, en cero)

# Ordenar por número de elemento (columna 1)
vec_to_sort = np.c_[lambnew2, el_new2, ep_new2, loggf_new2, damp_new2, disp2, infoeqwNew2]
indx_srt    = np.argsort(vec_to_sort[:, 1])
final_vec   = vec_to_sort[indx_srt, :]

formato = "%4.3f %2.1f %2.3f %2.3f  %2.3f %1.1f %3.1f"
np.savetxt(name_out + '.' + el_name + '.MOOG.dat', final_vec, fmt=formato)

print("Done! Output files saved:")
print("  ->", name_out + '.' + el_name + '.eqw.dat')
print("  ->", name_out + '.' + el_name + '.MOOG.dat')

# =============================================================================
# FIN DEL SCRIPT
# =============================================================================
