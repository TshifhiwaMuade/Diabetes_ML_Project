from dash import Dash, html

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Diabetes Risk Dashboard"),
    html.P("This is my first Dash app!")
])

if __name__ == "__main__":
    app.run(debug=True)

from dash import Dash, html, dcc, Input, Output
import pickle

app = Dash(__name__)
server = app.server

# Load model
model = pickle.load(open("../artifacts/model_xgb.pkl", "rb"))

app.layout = html.Div([
    html.H1("Diabetes Risk Prediction"),

    dcc.Input(id="age", type="number", placeholder="Enter Age"),
    dcc.Input(id="bmi", type="number", placeholder="Enter BMI"),

    html.Button("Predict", id="btn"),

    html.Div(id="output")
])

@app.callback(
    Output("output", "children"),
    Input("btn", "n_clicks"),
    Input("age", "value"),
    Input("bmi", "value")
)
def predict(n, age, bmi):
    if n is None:
        return ""

    data = [[age, bmi]]
    pred = model.predict(data)

    return f"Predicted Risk: {pred[0]}"

if __name__ == "__main__":
    app.run(debug=True)
