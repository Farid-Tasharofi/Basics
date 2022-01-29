from fileinput import filename
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.express as px
import seaborn as sns


username = 'FaridTasharofi'
api_key = 'xoKBbKUYiKAXOi50m9mT'

tls.set_credentials_file(username='FaridTasharofi',
                         api_key='xoKBbKUYiKAXOi50m9mT')


flights = sns.load_dataset("flights")
figure = px.scatter(x=range(10), y=range(10))
# Create a 3D scatter plot using flight data
figure = px.scatter_3d(flights, x='year', y='month', z='passengers', color='year',
                       opacity=0.7, width=400, height=200)


py.plot(figure, filename='sample', auto_open=False)
