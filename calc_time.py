import streamlit as st
import numpy as np
import rasterio
import ephem
import pytz
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import folium
from streamlit_folium import st_folium

def get_timezone_from_coordinates(lat, lon):
    """
    Get the timezone for given coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
    Returns:
        str: Timezone name (e.g., 'America/New_York')
    """
    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lat=lat, lng=lon)
    return timezone_name

def utc_to_local(date, timezone_name):
    """
    Convert UTC time to local time for a given timezone.
    
    Args:
        utc_time: ephem.Date object in UTC
        timezone_name: Timezone name (e.g., 'America/New_York')
    Returns:
        datetime: Local time as datetime object
    """
    time = pytz.utc.localize(date.datetime()).astimezone(pytz.timezone(timezone_name))

    return time

kV = lambda elev : 0.1066 + 0.12*np.exp(-elev/1500) # visual extinction
airmass = lambda Z, elev : (1 - (np.sin(Z)/(1+elev/6371000))**2)**-0.5 # airmass, Z is the altitude in radians, elev is the elevation in meters

data = pd.read_csv("data/KoonanDataCombined.csv")

Zvals = np.unique(data['Z'])
Tvals = np.unique(data['T'])
Hvals = np.unique(data['H'])

Bvals = np.reshape(data['B'], (len(Zvals), len(Tvals), len(Hvals)))

interim = RegularGridInterpolator((Zvals, Tvals, Hvals), np.log10(Bvals), bounds_error=False, fill_value=None)
sky_brightness = lambda x: 10 ** interim(x) # note: takes inputs in DEGREES. Z is the ALTITUDE, T is the AZIMUTH OFFSET, H is the Solar altitude. returns output in cd/m^2

def bright_limit(B): 
    """
    Calculate the limiting magnitude based on background brightness using the formula of Crumey (2014).
    
    Args:
        bcgd: Background brightness in cd/m^2
    Returns:
        float: Limiting magnitude
    """

    a1 = 6.112e-8; a2=-1.598e-7; a3=1.167e-7; a4=4.988e-4; a5=-3.014e-4
    I = (np.sqrt(a1*B**0.5 + a2*B**0.75 + a3*B) + a4*B**0.25 + a5*B**0.5)**2 # in lux
    return I

def I_to_mag(I):
    return -2.5*np.log10(I) - 13.99

def read_star_catalog(filename="data/catalog", mag_limit_l=-5.0, mag_limit_u=4.0):
    """
    Read the Yale Bright Star Catalog and return RA, Dec, and magnitude 
    for stars brighter than the specified magnitude limit.
    
    Based on the Yale Bright Star Catalog format:
    - RA is in columns 76-83 (HHMMSS.S format)  
    - Dec is in columns 84-91 (±DDMMSS format)
    - Visual magnitude is in columns 103-107
    
    Args:
        filename: Path to the catalog file
        mag_limit: Maximum magnitude to include (default 4.0)
    
    Returns:
        tuple: (ra_list, dec_list, mag_list) where:
            - ra_list: Right ascension in hours (list of floats)
            - dec_list: Declination in degrees (list of floats) 
            - mag_list: Visual magnitude (list of floats)
    """
    ra_list = []
    dec_list = []
    mag_list = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Skip empty lines or lines that are too short
                if len(line.strip()) < 110:
                    continue
                
                try:
                    # Extract RA (columns 76-83, format HHMMSS.S)
                    ra_str = line[75:83].strip()
                    if ra_str and len(ra_str) >= 6:
                        # Parse HHMMSS.S format
                        hours = int(ra_str[0:2])
                        minutes = int(ra_str[2:4])
                        seconds = float(ra_str[4:]) if len(ra_str) > 4 else 0.0
                        ra_hours = hours + minutes/60.0 + seconds/3600.0
                        ra_degrees = ra_hours * 15.0  # Convert to degrees
                    else:
                        continue
                    
                    # Extract Dec (columns 84-91, format ±DDMMSS)
                    dec_str = line[83:91].strip()
                    if dec_str and len(dec_str) >= 7:
                        # Parse ±DDMMSS format
                        sign = 1 if dec_str[0] != '-' else -1
                        deg_start = 1 if dec_str[0] in ['+', '-'] else 0
                        degrees = int(dec_str[deg_start:deg_start+2])
                        minutes = int(dec_str[deg_start+2:deg_start+4])
                        seconds = int(dec_str[deg_start+4:deg_start+6]) if len(dec_str) >= deg_start+6 else 0
                        dec_degrees = sign * (degrees + minutes/60.0 + seconds/3600.0)
                    else:
                        continue
                    
                    # Extract Visual Magnitude (columns 103-107)
                    mag_str = line[102:107].strip()
                    if mag_str and mag_str != '':
                        # Handle magnitude with possible leading sign
                        magnitude = float(mag_str)
                    else:
                        continue
                    
                    # Only include stars brighter than the magnitude limit
                    if mag_limit_l <= magnitude <= mag_limit_u:
                        ra_list.append(ra_degrees)
                        dec_list.append(dec_degrees)
                        mag_list.append(magnitude)
                        
                except (ValueError, IndexError) as e:
                    # Skip lines with parsing errors
                    continue
                    
    except FileNotFoundError:
        print(f"Catalog file {filename} not found")
        return [], [], []

    print(f"Successfully read {len(ra_list)} stars brighter than magnitude {mag_limit_u} and dimmer than {mag_limit_l}")
    starobjs = [ephem.FixedBody() for i in range(len(ra_list))]
    for i, (ra, dec, mag) in enumerate(zip(ra_list, dec_list, mag_list)):
        starobjs[i]._ra = np.radians(ra)  # Convert degrees to radians
        starobjs[i]._dec = np.radians(dec)
        starobjs[i].name = f"Star {i+1}"
        starobjs[i].mag = mag
    return starobjs

def light_pollution(lat, lon):
    """
    Calculate the light pollution at a given location using minimal memory.
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
    Returns:
        float: Light pollution value in cd/m^2
    """
    # filepath = "data/lp_cog.tif"

    shape = (17406, 43200)
    tile_size = 4096
        
    # Convert coordinates to pixel indices
    lat_idx = int((-lat + 85) * (shape[0] / 145))
    lon_idx = int((lon + 180) * (shape[1] / 360))
    
    # Ensure indices are within bounds
    lat_idx = max(0, min(lat_idx, shape[0] - 1))
    lon_idx = max(0, min(lon_idx, shape[1] - 1))

    tile_row = lat_idx // tile_size * tile_size
    tile_col = lon_idx // tile_size * tile_size

    filepath = f"data/lp_{tile_row}_{tile_col}.tif"

    local_row = lat_idx % tile_size
    local_col = lon_idx % tile_size

    with rasterio.open(filepath) as src:

        window = rasterio.windows.Window(local_col, local_row, 1, 1)
        lp = src.read(1, window=window)[0,0] # in mcd/m^2

        return 0.0 if np.isnan(lp) else lp*1e-3
    
def find_near_set(stars, d):
    """
    Efficiently find if the set of stars contains at least 3 stars closer than distance d.
    
    Args:
        stars: List of ephem star objects (already computed)
        d: Maximum distance threshold (in degrees)
    
    Returns:
        bool: True if at least 3 stars are within distance d of each other, False otherwise
    """
    n = len(stars)
    if n < 3:
        return False
    
    # For each star, find all neighbors within distance d
    for i in range(n):
        neighbors = []
        for j in range(n):
            if i != j and ephem.separation(stars[i], stars[j])*180/np.pi <= d:
                neighbors.append(j)
        
        # If star i has at least 2 neighbors, check if any pair of neighbors
        # are also within distance d of each other (forming a triangle)
        if len(neighbors) >= 2:
            for k in range(len(neighbors)):
                for l in range(k + 1, len(neighbors)):
                    if ephem.separation(stars[neighbors[k]], stars[neighbors[l]])*180/np.pi <= d:
                        return True  # Found a triangle of 3 stars
    
    return False
    
def nightfall(stars, obs, Blp=0.0):
    """
    Determines whether halakhic nightfall has occured. 

    Args:
        stars: List of star objects
        obs: Observer object
        Blp: Light pollution value at the observer's location, in cd/m^2

    Returns:
        bool: True if nightfall has occurred, False otherwise
    """
    # first, calculate the altitudes and azimuths of the stars
    alts = np.zeros(len(stars))
    azs = np.zeros(len(stars))
    mags = np.zeros(len(stars))
    for i, star in enumerate(stars):
        star.compute(obs)
        alts[i] = float(star.alt) # in radians
        azs[i] = float(star.az) # in radians
        mags[i] = float(star.mag) # in magnitudes

    # remove any stars that are too far below the horizon
    above_horizon = np.where(alts > 10*np.pi/180)
    alts = alts[above_horizon]
    azs = azs[above_horizon]
    mags = mags[above_horizon]

    # calculate the sun's position
    sun = ephem.Sun()
    sun.compute(obs)
    sun_alt = float(sun.alt)
    sun_az = float(sun.az)

    # calculate the twilight sky brightness at every star position
    vals = np.zeros((len(alts), 3))
    vals[:,0] = alts*180/np.pi # altitude in degrees
    vals[:,1] = np.abs(azs - sun_az)*180/np.pi # azimuth offset in degrees
    vals[:,1] = np.where(vals[:,1]>180, 360-vals[:,1], vals[:,1])
    vals[:,2] = sun_alt*180/np.pi # solar altitude in degrees
    Bvals = sky_brightness(vals)

    # correct for the altitude
    Bvals *= (1 - 10**(-0.4*kV(obs.elev) * airmass(np.pi/2-alts, obs.elev)))

    # add the light pollution value to the brightness values
    Bvals += Blp
    # add the night sky brightness
    Bvals += 4.329e-4

    Ivals = bright_limit(Bvals) # calculate the minimum brightness
    Ivals *= 10**(0.4*kV(obs.elev) * airmass(np.pi/2-alts, obs.elev)) # correct for attenuation
    lim_mags = I_to_mag(Ivals) - 1.0

    visible_stars = np.where(mags <= lim_mags)[0] # find the stars that are visible
    if len(visible_stars) >= 3:
        # check if there are at least 3 stars within 10 degrees of each other
        return find_near_set(np.array(stars)[above_horizon][visible_stars], 15)

    # otherwise, nightfall has not occurred
    return False

def halakhic_time(obs, stars, Blp=0.0):
    """
    Calculate the halakhic nightfall time for a given observer using a binary search through 100 minutes after sunset. 
    
    Args:
        obs: Observer object
        stars: List of star objects
        Blp: Light pollution value at the observer's location, in cd/m^2

    Returns:
        Date: Halakhic time as a Date object
    """
    # Start with the observer's local sunset time
    start_time = obs.next_setting(ephem.Sun())

    end_time = start_time + ephem.minute*100

    obs.date = end_time
    if not nightfall(stars, obs, Blp):
        print("WARNING: nightfall has not happened at the end!")
        return obs.date
    else:
        # perform a binary search
        time = start_time
        next_time = (start_time + end_time) / 2

        while np.abs(time - next_time) > ephem.second: # search down to a second
            time = next_time
            obs.date = time
            if nightfall(stars, obs, Blp):
                end_time = time
            else:
                start_time = time
            
            next_time = (start_time + end_time) / 2
        
        return obs.date
    
def sun_at_position(obs, h):
    """
    Get the next time that the sun is at a given altitude. The altitude must be negative, this is a binary search starting from sunset. 

    Args:
        obs: Observer object
        h: Altitude in radians

    Returns:
        ephem.Date: The next time the sun is at the given altitude
    """
    # Start with the observer's local sunset time
    sun = ephem.Sun()
    
    sun.compute(obs)
    obs.horizon = h + sun.radius

    return obs.next_setting(sun)

def calc_all_times(lat, lon, time, stars, Blp=0.0, elev=0.0):
    """
    Calculate all relevant times for a given observer and set of stars.

    Args:
        lat: Latitude of the observer
        lon: Longitude of the observer
        time: Time of interest
        stars: List of star objects
        Blp: Light pollution value at the observer's location, in cd/m^2
        elev: Elevation of the observer, in meters

    Returns:
        dict: A dictionary containing all relevant times
    """
    # assume the time is in local timezone
    # and convert it to UTC for calculations
    tz = get_timezone_from_coordinates(lat, lon)
    dt = pytz.timezone(tz).localize(time)  # Localize the datetime to the specified timezone
    date = ephem.Date(dt.astimezone(pytz.utc))  # Convert to UTC for ephem calculations

    obs = ephem.Observer()
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.elev = elev
    obs.date = date

    times = {}
    times['timezone'] = tz  # Store timezone name
    times['ht'] = utc_to_local(halakhic_time(obs, stars, Blp=0.0), tz)  # Halakhic time in local timezone, subtracted from the original time
    # times['ht_plus'] = utc_to_local(halakhic_time(obs, stars, error="plus", bckgd=False), tz) - dt # Halakhic time in local timezone
    # times['ht_minus'] = utc_to_local(halakhic_time(obs, stars, error="minus", bckgd=False), tz) - dt # Halakhic time in local timezone
    obs.date = date
    sunset = obs.next_setting(ephem.Sun())  # Sunset time in UTC
    times['sunset'] = utc_to_local(sunset, tz)  # Sunset time
    # Calculate approximate times for Rabbeinu Tam, Con, and Orth
    times['Tam_approx'] = utc_to_local(ephem.Date(sunset + ephem.minute * 72), tz)  # Rabbeinu Tam's approximation (72 minutes after sunset)
    obs.date = date
    times['Con_approx'] = utc_to_local(sun_at_position(obs, -7.5*np.pi/180), tz)  # Approximate time when the Sun is at -7.5 degrees altitude
    obs.date = date
    times['ht_lp'] = utc_to_local(halakhic_time(obs, stars, Blp=Blp), tz)

    return times

# Streamlit app
def main():
    # Configure page to reduce top margin
    st.markdown("""
    <style>
    .main > div {
        padding-top: 0.5rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    h1 {
        font-size: 2rem !important;
        margin-bottom: 0.0rem !important;
    }
    .header-link {
        text-align: right;
        margin-bottom: 1rem;
    }
    .header-link a {
        color: #1f77b4;
        text-decoration: none;
        font-size: 1.5rem;
    }
    .header-link a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header link
    st.markdown(
        "<div class='header-link'>"
        "<a href='https://www.astertaylor.com' target='_blank'>Home</a>"
        "</div>",
        unsafe_allow_html=True
    )
    
    st.title("Halakhic Time Calculator")

    # Location Settings in main page
    st.subheader("Location Settings")
    
    # Location input methods
    location_method = st.radio(
        "Choose location input method:",
        ["Map Selection", "City Search"],
        horizontal=True
    )
    
    if location_method == "Map Selection":
        st.markdown("**Click on the map to select your location:**")
        
        # Initialize session state for map coordinates if not exists
        if 'map_lat' not in st.session_state:
            st.session_state.map_lat = 31.7769  # Default to Jerusalem
        if 'map_lon' not in st.session_state:
            st.session_state.map_lon = 35.2345
        if 'map_key' not in st.session_state:
            st.session_state.map_key = 0
        
        # Create a Folium map
        m = folium.Map(
            location=[st.session_state.map_lat, st.session_state.map_lon],
            zoom_start=6,
            width='100%',
            height='400px'
        )
        
        # Add a marker for the current location
        folium.Marker(
            [st.session_state.map_lat, st.session_state.map_lon],
            popup=f"Selected Location\n({st.session_state.map_lat:.4f}, {st.session_state.map_lon:.4f})",
            tooltip="Click to confirm location",
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)
        
        # Display the map and capture click events
        map_data = st_folium(
            m,
            width=700,
            height=400,
            returned_objects=["last_clicked"],
            key=f"map_{st.session_state.map_key}"
        )
        
        # Update coordinates when map is clicked
        if map_data['last_clicked'] is not None:
            new_lat = map_data['last_clicked']['lat']
            new_lon = map_data['last_clicked']['lng']
            # Normalize longitude to -180 to +180 range
            normalized_lon = ((new_lon + 180) % 360) - 180
            if new_lat != st.session_state.map_lat or normalized_lon != st.session_state.map_lon:
                st.session_state.map_lat = new_lat
                st.session_state.map_lon = normalized_lon
                st.session_state.map_key += 1  # Force map refresh to update marker
                st.rerun()
        
        # Manual coordinate input for map mode
        col_map1, col_map2 = st.columns(2)
        with col_map1:
            manual_lat = st.number_input(
                f"Latitude ({abs(st.session_state.map_lat):.4f}° {'N' if st.session_state.map_lat >= 0 else 'S'})", 
                min_value=-90.0, 
                max_value=90.0, 
                value=st.session_state.map_lat, 
                step=1.0, 
                format="%.4f",
                key="manual_lat"
            )
        with col_map2:
            manual_lon = st.number_input(
                f"Longitude ({abs(st.session_state.map_lon):.4f}° {'E' if st.session_state.map_lon >= 0 else 'W'})", 
                min_value=-180.0, 
                max_value=180.0, 
                value=st.session_state.map_lon, 
                step=1.0, 
                format="%.4f",
                key="manual_lon"
            )
        
        # Update coordinates from manual input
        if manual_lat != st.session_state.map_lat or manual_lon != st.session_state.map_lon:
            # Normalize longitude to -180 to +180 range
            normalized_lon = ((manual_lon + 180) % 360) - 180
            st.session_state.map_lat = manual_lat
            st.session_state.map_lon = normalized_lon
            st.rerun()
        
        lat = st.session_state.map_lat
        lon = st.session_state.map_lon
        location_name = f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'} {abs(lon):.4f}° {'E' if lon >= 0 else 'W'}"
        
        # Show selected coordinates
        # st.info(f"Selected coordinates: {lat:.4f}, {lon:.4f}")
        
    else:
        # Predefined cities
        cities = {
            "Jerusalem": (31.7769, 35.2345),
            "New York": (40.7128, -74.0060),
            "London": (51.5074, -0.1278),
            "Ann Arbor": (42.2808, -83.7430),
            "Los Angeles": (34.0549, -118.2426),
            "Buenos Aires": (-34.6037, -58.3816),
            "Reykjavik": (64.1470, -21.9408),
            "Paris": (48.8575, 2.3514),
            "Alexandria": (31.2001, 29.9187)
        }
        
        col1, col2 = st.columns(2)
        with col1:
            selected_city = st.selectbox("Select a city:", list(cities.keys()))
            lat, lon = cities[selected_city]
            location_name = selected_city
        with col2:
            st.write(f"**Selected City:** {selected_city}")
        
        # Show map for selected city
        m_city = folium.Map(
            location=[lat, lon],
            zoom_start=6,
            width='100%',
            height='400px'
        )
        
        # Add a marker for the selected city
        folium.Marker(
            [lat, lon],
            popup=f"{selected_city}\n({lat:.4f}, {lon:.4f})",
            tooltip=f"Selected City: {selected_city}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m_city)
        
        # Display the city map (no interaction needed)
        st_folium(
            m_city,
            width=700,
            height=400,
            returned_objects=[],
            key=f"city_map_{selected_city}"
        )
        
        # Display coordinates in text boxes below the map (similar to map selection mode)
        col_city1, col_city2 = st.columns(2)
        with col_city1:
            st.number_input(
                f"Latitude ({abs(lat):.4f}° {'N' if lat >= 0 else 'S'})", 
                value=lat,
                format="%.4f",
                disabled=True,
                key=f"city_lat_{selected_city}"
            )
        with col_city2:
            st.number_input(
                f"Longitude ({abs(lon):.4f}° {'E' if lon >= 0 else 'W'})", 
                value=lon,
                format="%.4f",
                disabled=True,
                key=f"city_lon_{selected_city}"
            )
    
    # Elevation and Date inputs moved below the map
    col3, col4 = st.columns(2)
    with col3:
        elev = st.number_input("Elevation (meters)", min_value=0.0, value=0.0, step=1.0)
    with col4:
        selected_date = st.date_input("Select date", value=datetime.now().date())
    
    # Convert to datetime for calculations
    calc_date = datetime.combine(selected_date, datetime.min.time())
    
    st.divider()
    
    # Calculate button
    if st.button("Calculate Times", type="primary", use_container_width=True):
        with st.spinner("Loading star catalog and calculating times..."):
            try:
                # Load star catalog
                stars = read_star_catalog(mag_limit_l=2.0, mag_limit_u=4.0)
                
                if not stars:
                    st.error("Could not load star catalog. Please check if the catalog file exists.")
                    return
                
                # Get light pollution data
                light_poll = light_pollution(lat, lon)
                
                # Calculate times
                times = calc_all_times(lat, lon, calc_date, stars, Blp=light_poll, elev=elev)
                
                # Display results
                st.header(f"Halakhic Times for {location_name}")
                st.subheader(f"Date: {selected_date.strftime('%B %d, %Y')}")
                st.write(f"**Timezone:** {times['timezone']}")
                st.write(f"**Light Pollution:** {light_poll*1e3:.2f} mcd/m²")
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sunset Times")
                    st.write(f"**Sunset:** {times['sunset'].strftime('%H:%M:%S')}")
                    st.write(f"**Rabbeinu Tam (72 min):** {times['Tam_approx'].strftime('%H:%M:%S')}")
                    st.write(f"**R. Tukachinsky (-7.5°):** {times['Con_approx'].strftime('%H:%M:%S')}")
                
                with col2:
                    st.subheader("Halakhic Nightfall")
                    st.write(f"**Without Light Pollution:** {times['ht'].strftime('%H:%M:%S')}")
                    st.write(f"**With Light Pollution:** {times['ht_lp'].strftime('%H:%M:%S')}")
                    time_diff = times['ht_lp'] - times['ht']
                    st.write(f"**Light pollution delay:** {time_diff.total_seconds()/60:.1f} minutes")
                
                # Show difference
                # st.info(f"**Light pollution delay:** {time_diff.total_seconds()/60:.1f} minutes")
                
                # Additional information
                st.subheader("About These Times")
                st.write("""
                - **Sunset**: When the sun's disk disappears below the horizon
                - **Rabbeinu Tam**: Traditional fixed time of 72 minutes after sunset
                - **R. Tukachinsky (-7.5°)**: When the sun is 7.5° below the horizon
                - **Halakhic Nightfall**: When 3 medium stars become visible within 15° of each other
                - **Light pollution delay**: How much light pollution delays the appearance of stars
                """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check your inputs and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #FFFFFF; font-size: 0.9em; padding: 0 0;'>"
        "© Copyright 2025 Aster G. Taylor."
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()