import streamlit as st

st.set_page_config(page_title="Scouting Report App")

#st.sidebar.success("")
st.write('# Player Scouting Report App')
st.write('### Make the most out of season and game shooting data.')
st.write('\n\n')

st.write("""Generate impactful vizualisations. Compare players' data across games and seasons.
Take advantage of new insightful metrics and share your conclusions easily.
""")

st.write("""This App is divided into 2 tabs : Season and Game.""")

st.write("""#### In the Season Tab, you can visualize any player's  shooting averages in NBA seasons from 1996 onwards.""")

st.write("""Simply select a season and season type (Regular Season / PlayOffs), and then the player you want the shotmap of.


If you're new to this kind of data, I'll show you how to look at it with an example from 2024-25 LeBron :""")


with st.expander("Show More"):
    st.image("LeBron James shot chart 2024-25.png",width=600)

    st.write("""You can see hexagons all accross the court. They represent the shots taken by Lebron this year. The bigger the hexagon, the more shots taken from this exact zone. 
    The color represents the efficiency relative to league average. Navy blue is -10% relative to league average or below, red is +10% and above, yellow is average. 
    \n\n
    In this example, you can see Lebron's athleticism and touch make him a force downhill, with both volume and great efficiency at the rim. However, his 3 pointer still needs work, and shows better percentages going right than going left. 
    """)
    st.write("""This is how you extract value from these graphs. Save the 2024-25 season graph, check the 2023-24 one and compare zones and percentages. What do you see ? How does his shot regime change ? Did he take less shots from midrange and more 3 ? 
    Did his percentages improve ? What could be the reason he has better percentages going right than left ?
    """)

st.write("""#### In the Game Tab, you can visualize single game performances from players in 2024-25 (Regular Season + PlayOffs).""")

st.write("""Use this tab to compare season averages with one time performance. Did the player missed shots he usually makes ? Was he on fire that day ? How does his percentages compare with his averages ? Was it the defense that slowed him down or is it just 3P variance ?""")
st.write('\n\n\n\n\n\n\n\n\n\n\n\n')
st.write('\n\n\n\n\n\n\n\n\n\n\n\n')
st.write('\n\n\n\n\n\n\n\n\n\n\n\n')
st.write('\n\n\n\n\n\n\n\n\n\n\n\n')

st.write("""
Disclaimer : This model and graphs do not take defense into account. 
A player can shoot 70% at the rim for the season and put up mediocre attemps against a well organised PlayOff defense, his xPTS will still account for the RS numbers.
You have to take this into account when using these numbers to evaluate performances. Stats don't tell the full story, always back up your claims with film. The link to game footage is to be found below game shotchart graph.
""")
twitter_url = "https://x.com/ZieseI"
st.write("""For more visualizations like that, follow me on twitter [here](%s)""" % twitter_url)