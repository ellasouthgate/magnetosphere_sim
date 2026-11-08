# Magnetosphere / aurora simulation

Charged particles in Earth's dipole field (Boris pusher), then a tilted Earth
plus a simple solar field and aurora colours.

    python run_dipole.py      # dipole, magnetopause, particles
    python run_extended.py    # extended model, figures ext_*
    python -m pytest          # tests

Parameters are in `config.py`, constants are in `constants.py`.

## Coordinates

Origin at the centre of Earth. The Sun is on -x (left of the x-z plots), z is
north at zero tilt and y = z cross x, so +y is dawn. Units are metres and tesla
in the code, plots use R_E = 6371 km.

## Field

    B_total = B_earth + w_sun * B_sun

Earth is a dipole tilted 23.4 deg towards the Sun (21 June). The Sun is a second
dipole at x = -1000 R_E, so its field is nearly uniform near Earth. Its strength
is set by r_balance, the distance where Earth's equatorial field equals the
solar field (default 10 R_E, about 31 nT, real IMF is about 5 nT).

Field lines are closed (Earth at both ends), open (one end) or solar (neither).
For the default model the nulls are at (-9.64, 0, -2.65) and (9.81, 0, 2.75) R_E.

With no tilt and a uniform field the open/closed boundary is at

    cos^2(lat) = (3/(2 r_b)) / (1 + 1/(2 r_b^3))

which is 67.22 deg for r_b = 10. The code gets 67.13 (day) and 67.31 (night).

## Aurora

Everything is in `aurora_colour.py`. Exponential N2 and O atmospheres, Chapman
style energy deposition and collisional quenching give the emission of each line
vs height. Peaks are 122 km (557.7 nm), 214 km (630.0 nm) and 108 km (427.8 and
670 nm).

Colours use the CIE 1931 functions (`data/ciexyz31_1.csv`, from CVRL), then
XYZ to sRGB with OKLCh gamut mapping and a log brightness scale.

The glow is a Gaussian band around the open/closed boundary. Heights are
exaggerated on the plots.

## Limitations

- no solar wind, the solar field is a static dipole
- single scale heights and one electron energy
- only four emission lines, strengths tuned to literature ratios
- glow is drawn straight above the footpoint
