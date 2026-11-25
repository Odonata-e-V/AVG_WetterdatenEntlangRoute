
import zipfile
import pandas as pd
import numpy as np

#Zip einlesen
ZIP_FILE = "ERA5_Corridor_Results_July.zip"

HEIGHT_MIN = 2       # m
HEIGHT_MAX = 3000    # m
HEIGHT_STEP = 1      # m

# Konstanten
g = 9.80665          # m/s²
R = 287.05           # J/(kg K)

# Sutherland-Konstanten für Luft
MU_0 = 1.716e-5      # Pa s
T_0 = 273.15         # K
S = 110.4            # K


with zipfile.ZipFile(ZIP_FILE, "r") as archive:

    single = pd.read_csv(
        archive.open(
            "ERA5_Corridor_Results/"
            "ERA5_corridor_single_levels.csv"
        )
    )

    pressure = pd.read_csv(
        archive.open(
            "ERA5_Corridor_Results/"
            "ERA5_corridor_pressure_levels.csv"
        )
    )


# Zeitstempel vereinheitlichen
single["time"] = pd.to_datetime(single["time"])
pressure["time"] = pd.to_datetime(pressure["time"])


#Höhenraster

heights = np.arange(
    HEIGHT_MIN,
    HEIGHT_MAX + HEIGHT_STEP,
    HEIGHT_STEP
)

temperature_profiles = []
pressure_profiles = []


# zeitliche Interpolation

for _, row in single.iterrows():

    time = row["time"]

    # Oberflächenwerte
    T_surface = row["t2m"]
    p_surface = row["sp"]
    p_msl = row["msl"]

    # Ungefähre Geländehöhe aus dem
    # reduzierten und tatsächlichen Luftdruck
    terrain_height = (
        R * T_surface / g
        * np.log(p_msl / p_surface)
    )

    # Oberflächenpunkt auf 2 m Höhe
    p_2m = p_surface * np.exp(
        -g * (HEIGHT_MIN - terrain_height)
        / (R * T_surface)
    )

    # Drucklevel des jeweiligen Zeitpunkts
    levels = pressure[
        pressure["time"] == time
    ].copy()

    levels = levels.sort_values(
        "pressure_level",
        ascending=False
    )

    # Geopotential in geometrische Näherungshöhe umrechnen
    level_heights = (
        levels["z"].to_numpy() / g
        - terrain_height
    )

    level_temperatures = levels["t"].to_numpy()

    # Druck von hPa in Pa umrechnen
    level_pressures = (
        levels["pressure_level"].to_numpy() * 100
    )

    # Höhenpunkte zusammensetzen
    profile_heights = np.concatenate((
        [HEIGHT_MIN],
        level_heights
    ))

    profile_temperatures = np.concatenate((
        [T_surface],
        level_temperatures
    ))

    profile_pressures = np.concatenate((
        [p_2m],
        level_pressures
    ))

    # Nach Höhe sortieren
    order = np.argsort(profile_heights)

    profile_heights = profile_heights[order]
    profile_temperatures = profile_temperatures[order]
    profile_pressures = profile_pressures[order]

    # Temperatur linear interpolieren
    T_interp = np.interp(
        heights,
        profile_heights,
        profile_temperatures
    )

    # Druck logarithmisch interpolieren
    ln_p_interp = np.interp(
        heights,
        profile_heights,
        np.log(profile_pressures)
    )

    p_interp = np.exp(ln_p_interp)

    temperature_profiles.append(T_interp)
    pressure_profiles.append(p_interp)


# Zeitliche Mittelwerte


T_mean = np.mean(
    temperature_profiles,
    axis=0
)

p_mean = np.mean(
    pressure_profiles,
    axis=0
)

# Dynamische Viskosität mittels Sutherland-Gleichung

mu = MU_0 * (
    (T_mean / T_0) ** 1.5
    * (T_0 + S) / (T_mean + S)
)

# Ergebnisse speichern


results = pd.DataFrame({

    "height_m": heights,

    "temperature_K": T_mean,

    "temperature_C": T_mean - 273.15,

    "pressure_Pa": p_mean,

    "pressure_hPa": p_mean / 100,

    "dynamic_viscosity_Pa_s": mu

})

results.to_csv(
    "atmosphere_profile.csv",
    index=False
)

print(results.to_string(index=False))

print("\nBerechnung abgeschlossen!")
print("Datei: atmosphere_profile.csv")
