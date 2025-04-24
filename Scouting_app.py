import streamlit as st

st.set_page_config(page_title="Scouting Report App",layout='wide')

st.sidebar.success("Select a use case.")
st.write('# Players scouting report app')


st.write("""
- Hexagons are drawn in each zone of the court. Their size and color depend on the selected player's performance shooting in these zones.
Size represents volume (the more a player shoots from there, the bigger the hexagon), and color is quality (red = better than average, blue = worse)

- TS% : Measure of a player's efficiency. Dependant on PTS, FTA and FGA.

- xPTS : How many points a player "should have scored" based on his shot selection and season averages in these zones.

- Placeholder : images des 2 apps pour donner une idée         
         """)
st.image("legend.png")