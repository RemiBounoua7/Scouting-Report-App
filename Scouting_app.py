import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import RegularPolygon,Circle, Rectangle, Arc
from pathlib import Path
from itertools import product
from urllib.request import urlopen
import tarfile
from typing import Union, Sequence, Optional, List
from io import BytesIO, TextIOWrapper
import csv
import streamlit as st
import matplotlib
from nba_api.stats.endpoints import shotchartdetail,playergamelog
from nba_api.stats.static import teams
from PIL import Image

matplotlib.use('Agg')



@st.cache_data
def load_nba_data(path: Union[Path, str] = Path.cwd(),
                  seasons: Union[Sequence, int] = range(1996, 2024),
                  data: Union[Sequence, str] = ("datanba", "nbastats", "pbpstats",
                                                "shotdetail", "cdnnba", "nbastatsv3"),
                  seasontype: str = 'rg',
                  league: str = 'nba',
                  untar: bool = False,
                  in_memory: bool = False,
                  use_pandas: bool = True) -> Optional[Union[List, pd.DataFrame]]:
    """
    Loading a nba play-by-play dataset from github repository https://github.com/shufinskiy/nba_data

    Args:
        path (Union[Path, str]): Path where downloaded file should be saved on the hard disk. Not used if in_memory = True
        seasons (Union[Sequence, int]): Sequence or integer of the year of start of season
        data (Union[Sequence, str]): Sequence or string of data types to load
        seasontype (str): Part of season: rg - Regular Season, po - Playoffs
        league (str): Name league: NBA or WNBA
        untar (bool): Logical: do need to untar loaded archive. Not used if in_memory = True
        in_memory (bool): Logical: If True dataset is loaded into workflow, without saving file to disk
        use_pandas (bool): Logical: If True dataset is loaded how pd.DataFrame, else List[List[str]]. Ignore if in_memory=False

    Returns:
        Optional[pd.DataFrame, List]: If in_memory=True and use_pandas=True return dataset how pd.DataFrame.
        If use_pandas=False return dataset how List[List[str]]
        If in_memory=False return None
    """
    if isinstance(path, str):
        path = Path(path).expanduser()
    if isinstance(seasons, int):
        seasons = (seasons,)
    if isinstance(data, str):
        data = (data,)

    if (len(data) > 1) & in_memory:
        raise ValueError("Parameter in_memory=True available only when loading a single data type")

    if seasontype == 'rg':
        need_data = tuple(["_".join([data, str(season)]) for (data, season) in product(data, seasons)])
    elif seasontype == 'po':
        need_data = tuple(["_".join([data, seasontype, str(season)]) \
                           for (data, seasontype, season) in product(data, (seasontype,), seasons)])
    else:
        need_data_rg = tuple(["_".join([data, str(season)]) for (data, season) in product(data, seasons)])
        need_data_po = tuple(["_".join([data, seasontype, str(season)]) \
                              for (data, seasontype, season) in product(data, ('po',), seasons)])
        need_data = need_data_rg + need_data_po
    if league.lower() == 'wnba':
        need_data = ['wnba_' + x for x in need_data]

    check_data = [file + ".csv" if untar else "tar.xz" for file in need_data]
    not_exists = [not path.joinpath(check_file).is_file() for check_file in check_data]

    need_data = [file for (file, not_exist) in zip(need_data, not_exists) if not_exist]

    with urlopen("https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt") as f:
        v = f.read().decode('utf-8').strip()

    name_v = [string.split("=")[0] for string in v.split("\n")]
    element_v = [string.split("=")[1] for string in v.split("\n")]

    need_name = [name for name in name_v if name in need_data]
    need_element = [element for (name, element) in zip(name_v, element_v) if name in need_data]

    if in_memory:
        if use_pandas:
            table = pd.DataFrame()
        else:
            table = []
    for i in range(len(need_name)):
        with urlopen(need_element[i]) as response:
            if response.status != 200:
                raise Exception(f"Failed to download file: {response.status}")
            file_content = response.read()
            if in_memory:
                with tarfile.open(fileobj=BytesIO(file_content), mode='r:xz') as tar:
                    csv_file_name = "".join([need_name[i], ".csv"])
                    csv_file = tar.extractfile(csv_file_name)
                    if use_pandas:
                        table = pd.concat([table, pd.read_csv(csv_file)], axis=0, ignore_index=True)
                    else:
                        csv_reader = csv.reader(TextIOWrapper(csv_file, encoding="utf-8"))
                        for row in csv_reader:
                            table.append(row)
            else:
                with path.joinpath("".join([need_name[i], ".tar.xz"])).open(mode='wb') as f:
                    f.write(file_content)
                if untar:
                    with tarfile.open(path.joinpath("".join([need_name[i], ".tar.xz"]))) as f:
                        f.extract("".join([need_name[i], ".csv"]), path)

                    path.joinpath("".join([need_name[i], ".tar.xz"])).unlink()
    if in_memory:
        return table
    else:
        return None

def calculate_hexbin_stats(df, x_col='LOC_X', y_col='LOC_Y', shot_col='SHOT_MADE_FLAG',
                           grid_size=20, extent=(-250, 250, -50, 300)):
    """
    Calculate hexbin stats for a given set of shots.
    
    Returns:
        stats_dict (dict): Dictionary with keys:
            'x_centers': ndarray of hexagon center x-coordinates.
            'y_centers': ndarray of hexagon center y-coordinates.
            'volumes': ndarray of raw shot counts per hexagon.
            'norm_volumes': ndarray of normalized shot volumes (0-1) in this dataset.
            'fg_percentages': ndarray of FG% per hexagon.
    """
    # Create hexbin for shot volume (count per bin)
    hb_counts = plt.hexbin(
        df[x_col], df[y_col],
        gridsize=grid_size,
        extent=extent,
        mincnt=0
    )
    volumes = np.nan_to_num(hb_counts.get_array(), nan=0)

    x_centers, y_centers = hb_counts.get_offsets().T
    plt.close()

    # Calculate total attempts per bin (using ones)
    hb_attempts = plt.hexbin(
        df[x_col], df[y_col],
        C=np.ones(len(df)),
        gridsize=grid_size,
        extent=extent,
        reduce_C_function=np.sum,
        mincnt=0
    )
    total_attempts = np.nan_to_num(hb_attempts.get_array(), nan=0)
    plt.close()
    
    # Calculate total successes per bin (using shot made flag)
    hb_successes = plt.hexbin(
        df[x_col], df[y_col],
        C=df[shot_col],
        gridsize=grid_size,
        extent=extent,
        reduce_C_function=np.sum,
        mincnt=0
    )

    total_successes = np.nan_to_num(hb_successes.get_array(), nan=0)
    plt.close()
    
    # Compute FG% per bin
    fg_percentages = np.divide(
        total_successes, 
        total_attempts, 
        out=np.zeros_like(total_successes, dtype=float), 
        where=total_attempts > 0
    )
    fg_percentages = [i if i is not None else 0 for i in fg_percentages]   

    # Normalize shot volumes (for this dataset)
    norm_volumes = volumes / volumes.max() if volumes.max() > 0 else 0


    stats_dict = {
        'x_centers': x_centers,
        'y_centers': y_centers,
        'volumes': volumes,
        'norm_volumes': norm_volumes,
        'fg_percentages': fg_percentages
    }
    return stats_dict

def compare_player_to_global(df, player_name, x_col='LOC_X', y_col='LOC_Y', shot_col='SHOT_MADE_FLAG',
                             grid_size=20, extent=(-250, 250,-50,300)):
    """
    Compare a player's hexbin stats to the global (average) stats.
    
    Returns:
        comparison (list of dict): Each entry corresponds to a hexagon (bin) that exists in the global data.
            For each bin, it includes:
                - x_center, y_center
                - global_volume, global_fg
                - player_volume, player_fg (or np.nan if player has no shots in that bin)
                - diff_volume (player_volume - global_volume) or ratio as desired
                - diff_fg (player_fg - global_fg) or ratio as desired
    """
    # Calculate global stats (all shots)
    global_stats = calculate_hexbin_stats(df, x_col, y_col, shot_col, grid_size, extent)

    # Calculate player stats (filtered)
    player_df = df[df['PLAYER_NAME'] == player_name]

    player_stats = calculate_hexbin_stats(player_df, x_col, y_col, shot_col, grid_size, extent)

    # Prepare output by matching bins using the hexagon centers.
    # We assume that the global hexbin grid covers more bins; for each global bin, try to find a matching player bin.
    comparison = []
    for i, (gx, gy, gvol, gfg) in enumerate(zip(global_stats['x_centers'], 
                                                  global_stats['y_centers'], 
                                                  global_stats['volumes'], 
                                                  global_stats['fg_percentages'])):
        # Try to find a matching bin in player's stats (using a tolerance in case of floating point differences)
        match_idx = None
        for j, (px, py) in enumerate(zip(player_stats['x_centers'], player_stats['y_centers'])):
            if np.allclose([gx, gy], [px, py], atol=1e-6):
                match_idx = j
                break
        if match_idx is not None:
            pvol = player_stats['volumes'][match_idx]
            pfg = player_stats['fg_percentages'][match_idx]
        else:
            pvol = 0
            pfg = 0


        # For the comparison you can compute differences or ratios. Here we compute differences.
        diff_volume = pvol - gvol if not np.isnan(pvol) else np.nan
        diff_fg = pfg - gfg if not np.isnan(pfg) else np.nan
        
        comparison.append({
            'x_center': gx,
            'y_center': gy,
            'global_volume': gvol,
            'global_fg': gfg,
            'player_volume': pvol,
            'player_fg': pfg,
            'diff_volume': diff_volume,
            'diff_fg': diff_fg
        })
    comparison = [
    {key: (0 if np.isnan(value) else value) for key, value in comp.items()}
    for comp in comparison
]
    return comparison

def draw_courts(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get the current one
    if ax is None:
        ax = plt.gca()
         
    # Create the basketball hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle(( - 30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box of the paint, width=16ft, height=19ft
    outer_box = Rectangle(( - 80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    # Create the inner box of the paint, width=12ft, height=19ft
    inner_box = Rectangle(( - 60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle(( - 220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle(( + 220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                        bottom_free_throw, restricted, corner_three_a,
                        corner_three_b, three_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        court_outer_lines = Rectangle(( - 250, -47.5), 500, 380, linewidth=lw, color=color, fill=False)
        court_elements.append(court_outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def plot_comparison(comparison,ax):
    """
    Plot the comparison statistic by drawing hexagons.
    
    Args:
        comparison (list of dict): Output from compare_player_to_global.
        draw_court (function): Function to draw the basketball court.
        stat (str): Which statistic to plot ('diff_fg' or 'diff_volume').
        title_suffix (str): Suffix for the plot title.
    """
    
    # Choose colormap and normalization based on statistic.
    custom_colors = ['#192BC2', '#78C0E0', '#FCDC4D', '#CB793A', '#9A031E']

    cmap = ListedColormap(custom_colors)
    fg_intervals = np.linspace(-.2, .2, len(custom_colors) + 1)
    
        
    for bin_stat in comparison:
        x = bin_stat['x_center']
        y = bin_stat['y_center']

        # Get the player's FG% relative to league average
        value = bin_stat['diff_fg']
        
        size = bin_stat['player_volume'] if not np.isnan(bin_stat['player_volume']) else 0
        max_hex_size = 10.5
        color_idx = np.digitize(bin_stat['diff_fg'], fg_intervals) - 1  # Get the index for the color
        radius = min(size , max_hex_size) if size>1 else 0
        hexagon = RegularPolygon((x, y), numVertices=6, radius=radius, orientation=np.radians(0),
                                 facecolor=cmap(color_idx) if not np.isnan(value) else 'gray', 
                                 alpha=0.75, edgecolor='k')
        ax.add_patch(hexagon)    

def get_hex(comparison,x,y):
    index=0
    res=((comparison[0]['x_center']-x)**2+(comparison[0]['y_center']-y)**2)**.5
    for k in range(1,len(comparison)):
        if ((comparison[k]['x_center']-x)**2+(comparison[k]['y_center']-y)**2)**.5 < res:
            if not np.isnan(comparison[k]['player_fg']) and comparison[k]['player_fg']!="masked":
                index=k
                res=((comparison[k]['x_center']-x)**2+(comparison[k]['y_center']-y)**2)**.5
    return index

def get_expected_pts(comparison,shotchart,season_df,game_df):
    x_pts=0

    for index,(x,y,res,value) in shotchart.iterrows():

        value=int(value)
        hex = get_hex(comparison,x,y)
        x_pts += value*comparison[hex]['player_fg']


    x_pts += game_df['FTA'].values[0] * season_df['FT_PCT'].mean()
    return round(x_pts,1)


def get_player_season_averages(season_df):

    pts = round(season_df['PTS'].mean(),1)
    _2ptFG_PCT = str(int(100*round(season_df['FG_PCT'].mean(),2)))
    _3ptFG_PCT = str(int(100*round(season_df['FG3_PCT'].mean(),2)))
    _FT_PCT = str(int(100*round(season_df['FT_PCT'].mean(),2)))
    minutes = str(round(season_df['MIN'].mean(),1))
    
    TS_PCT = str(round(50*(pts)/(season_df['FGA'].mean()+0.44*season_df['FTA'].mean()),1))

    stats_str =[minutes,str(pts),_2ptFG_PCT,_3ptFG_PCT,_FT_PCT,TS_PCT]
     

    return stats_str

def get_player_game_stats(game_df):
    pts = game_df['PTS'].values[0]
    x_pts = str(get_expected_pts(comparison, game_shotchart,selected_player_season_df,selected_game_df))

    minutes = str(game_df['MIN'].values[0])

    TS_PCT = str(round(50*(pts)/(game_df['FGA'].values[0]+.44*game_df['FTA'].values[0]),1))

    stats_str=[minutes,str(pts),x_pts, f"{game_df['FGM'].values[0]}/{game_df['FGA'].values[0]}",f"{game_df['FG3M'].values[0]}/{game_df['FG3A'].values[0]}",f"{game_df['FTM'].values[0]}/{game_df['FTA'].values[0]}",TS_PCT]
    
    return stats_str



st.set_page_config(page_title="Scouting Report App",layout='wide')

st.write('# Players scouting report app')


df = load_nba_data(
    seasons=2024,
    data="shotdetail",
    in_memory=True
)
df=df[['PLAYER_NAME','LOC_X','LOC_Y','SHOT_MADE_FLAG','PLAYER_ID']]
# Reverse left-right because of data gathering from the NBA is the other way around.
df['LOC_X'] = df['LOC_X'].apply(lambda x:-x)

selected_player = st.selectbox(
    "Select the player",
    sorted(df['PLAYER_NAME'].unique()),
    index=487,
    placeholder="Select a player ...")

selected_player_id = df[df['PLAYER_NAME']==selected_player].iloc[0]['PLAYER_ID']

selected_player_regular_season_df = playergamelog.PlayerGameLog(player_id=selected_player_id, season='2024-25').get_data_frames()[0]
selected_player_playoffs_df = playergamelog.PlayerGameLog(player_id=selected_player_id, season='2024-25',season_type_all_star="Playoffs").get_data_frames()[0]
selected_player_season_df = pd.concat([selected_player_playoffs_df,selected_player_regular_season_df])
selected_player_season_df['Matchup + Date'] = selected_player_season_df['MATCHUP'].apply(lambda x: x[4:]) + " - " + selected_player_season_df['GAME_DATE']

selected_game_name = st.selectbox(
    "Pick a Game",
    selected_player_season_df['Matchup + Date'],
    index=0,
    placeholder="Select game ...",
)

selected_game_df = selected_player_season_df[selected_player_season_df['Matchup + Date']==selected_game_name]
selected_game_id = selected_game_df['Game_ID'].values[0]

Team_ID = teams.find_team_by_abbreviation(selected_game_df['MATCHUP'].values[0][:3])['id']

rs_game_shotchart = shotchartdetail.ShotChartDetail(
    player_id=selected_player_id,
    team_id=Team_ID,
    game_id_nullable=selected_game_id,
    context_measure_simple='FGA',
).get_data_frames()[0]
po_game_shotchart = shotchartdetail.ShotChartDetail(
    player_id=selected_player_id,
    team_id=Team_ID,
    game_id_nullable=selected_game_id,
    context_measure_simple='FGA',
    season_type_all_star='Playoffs'
).get_data_frames()[0]
game_shotchart = pd.concat([po_game_shotchart,rs_game_shotchart])

game_shotchart=game_shotchart[['LOC_X','LOC_Y','SHOT_MADE_FLAG', 'SHOT_TYPE']]
game_shotchart['SHOT_TYPE'] = game_shotchart['SHOT_TYPE'].apply(lambda x: x[0])
game_shotchart['LOC_X'] = game_shotchart['LOC_X'].apply(lambda x:-x)

# Don't ask me why, but the hexbins density get plot on the last ax. So we circumvent that by creating empty graphs (in a lower row not to mess with our courts length) to plot it in.
figure, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1,0]}, figsize=(8,3))
draw_courts(ax1,outer_lines=True)
draw_courts(ax2,outer_lines=True)

ax1.set_xlim(-251,251)
ax1.set_ylim(-50,335)
ax1.set_axis_off()
ax1.set_title(f"{selected_player} Shot Chart (2024-25)",fontdict={'fontsize': 7})

ax2.set_xlim(-251,251)
ax2.set_ylim(-50,335)
ax2.set_axis_off()
ax2.set_title(f"{selected_player} {selected_game_name} ({selected_game_df['WL'].values[0]})",fontdict={'fontsize': 7})

ax3.set_axis_off()


player_photo_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{selected_player_id}.png?imwidth=1040&imheight=760"
player_photo=Image.open(urlopen(player_photo_url))



comparison = compare_player_to_global(df, selected_player)
plot_comparison(comparison, ax=ax1)
for index,(x,y,res,value) in game_shotchart.iterrows():
    if res==1:
        #ax2.scatter(x,y,color='green',marker='o')
        ax2.scatter(x,y,facecolors='none', edgecolors='g',zorder=10)
    else:
        ax2.scatter(x,y,color='red', marker="x",zorder=9)

#st.write(get_player_season_averages(selected_player_season_df))
season_stats = get_player_season_averages(selected_player_season_df)
game_stats = get_player_game_stats(selected_game_df)
#figure.text(0.3, 0.03, season_stats, horizontalalignment="center",fontdict={'fontsize': 7})
#figure.text(.685, 0.03, game_stats, horizontalalignment="center",fontdict={'fontsize': 7})
season_labels = ["MIN", "PTS", "FG%","3FG%","FT%","TS%"]
game_labels = ["MIN", "PTS","xPTS", "FG","3FG","FT","TS%"]


for i, (num, label) in enumerate(zip(season_stats, season_labels)):
    # Calculate x position for each pair
    x = -170 + 65*i 

    ax1.text(x, -70, num, ha='center', va='center', fontsize=9, color='black', fontweight='bold')
    ax1.text(x, -85, label, ha='center', va='center', fontsize=5, color='grey', fontweight='medium')

for j, (num, label) in enumerate(zip(game_stats, game_labels)):
    # Calculate x position for each pair
    x = -170 + 65*j 

    ax2.text(x, -70, num, ha='center', va='center', fontsize=9, color='black', fontweight='bold')
    ax2.text(x, -85, label, ha='center', va='center', fontsize=5, color='grey', fontweight='medium')


image_ax = figure.add_axes([0.375, 0.111, 0.23, 0.23])  # [x, y, width, height]
image_ax.imshow(player_photo)
image_ax.axis("off")  # Hide axes for the image

# Adjust layout to prevent overlap
plt.tight_layout()

st.pyplot(figure)




# Button to save and download the figure
buffer = BytesIO()
figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")  # Save the figure to the buffer
buffer.seek(0)  # Reset the buffer position

game_video_link = f"https://www.nba.com/stats/events?CFID=&CFPARAMS=&ContextMeasure=FGA&EndPeriod=0&EndRange=28800&GameID={selected_game_id}&PlayerID={selected_player_id}&Season=2024-25&TeamID={Team_ID}&flag=3&sct=plot"
season_video_link = f"https://www.nba.com/stats/events?CFID=&CFPARAMS=&ContextMeasure=FGA&EndPeriod=0&EndRange=28800&PlayerID={selected_player_id}&Season=2024-25&TeamID={Team_ID}&flag=3&sct=plot"


c1,c2,c3 = st.columns(3)
with st.container():
    c1.write("[Season Film](%s)" % season_video_link)
    c2.download_button(
        label="Save Graphs",
        data=buffer,
        file_name=f"{selected_player} shot chart {selected_game_name}.png",
    )
    c3.write("[Game Film](%s)" % game_video_link)


with st.expander("Legend"):
    st.write("""
- Hexagons are drawn in each zone of the court. Their size and color depend on the selected player's performance shooting in these zones.
Size represents volume (the more a player shoots from there, the bigger the hexagon), and color is quality (red = better than average, blue = worse)

- TS% : Measure of a player's efficiency. Dependant on PTS, FTA and FGA.

- xPTS : How many points a player "should have scored" based on his shot selection and season averages in these zones.
""")
