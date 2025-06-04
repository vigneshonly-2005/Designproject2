import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
import torch

from prophet import Prophet

from openai import OpenAI
from google.colab import userdata
import gradio as gr

class BIAssistant:
    def _init_(self):
        self.data = None
        self.column_map = {
            'date': None,
            'product': None,
            'region': None,
            'sales': None,
            'customer_age': None
        }
        self.stats_summary = None
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=userdata.get('nvidiaAPI')
        )

    def load_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path.name)
            self._detect_columns()

            if self.column_map['date']:
                self.data[self.column_map['date']] = pd.to_datetime(
                    self.data[self.column_map['date']], errors='coerce'
                )

            return "Data loaded successfully!", self._get_data_preview()
        except Exception as e:
            return f"Error loading data: {str(e)}", None

    def _detect_columns(self):
        for col in self.data.columns:
            col_lower = col.lower()
            for expected_col in self.column_map.keys():
                if expected_col in col_lower:
                    self.column_map[expected_col] = col

    def _get_data_preview(self):
        return f"""
        <h3>Data Preview</h3>
        <p>Rows: {len(self.data)}, Columns: {len(self.data.columns)}</p>
        {self.data.head().to_html()}
        """

    def initialize_system(self):
        try:
            self.stats_summary = self._generate_statistical_summaries()
            return "System initialized successfully!"
        except Exception as e:
            return f"Initialization failed: {str(e)}"

    def _generate_statistical_summaries(self):
        summary = "Business Data Summary:\n"
        product_col = self.column_map['product']
        sales_col = self.column_map['sales']
        date_col = self.column_map['date']

        if product_col and sales_col:
            try:
                top_products = self.data.groupby(product_col)[sales_col].sum().nlargest(5)
                summary += f"\nTop 5 Products by Sales:\n{top_products.to_string()}\n"
            except:
                summary += "\nProduct analysis unavailable\n"

        if date_col and sales_col:
            try:
                monthly_sales = self.data.groupby(pd.Grouper(key=date_col, freq='ME'))[sales_col].sum()
                summary += f"\nLast 3 Months Sales:\n{monthly_sales.tail(3).to_string()}\n"
            except:
                summary += "\nSales trend analysis unavailable\n"

        return summary

    def ask_question(self, question):
        try:
            if not self.stats_summary:
                return "System not initialized. Please load data and initialize first."

            system_message = (
                "You are a business intelligence assistant. Use the provided business data "
                "to answer user queries with clear insights.\n\n"
                + self.stats_summary
            )

            completion = self.client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-super-49b-v1",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )

            return completion.choices[0].message.content.strip().replace('*', '')

        except Exception as e:
            return f"Error answering question: {str(e)}"

    def generate_custom_plot(self, chart_type="Line", mode="trends"):
        try:
            date_col = self.column_map['date']
            sales_col = self.column_map['sales']
            product_col = self.column_map['product']

            plt.figure(figsize=(10, 6))

            if mode == "trends" and date_col and sales_col:
                data = self.data.groupby(pd.Grouper(key=date_col, freq='ME'))[sales_col].sum().dropna()
                x = data.index
                y = data.values
            elif mode == "product" and product_col and sales_col:
                data = self.data.groupby(product_col)[sales_col].sum().nlargest(10)
                x = data.index
                y = data.values
            else:
                return None

            if chart_type == "Line":
                plt.plot(x, y)
            elif chart_type == "Bar":
                plt.bar(x, y)
            elif chart_type == "Pie":
                plt.pie(y, labels=x, autopct='%1.1f%%')
                plt.axis('equal')
            elif chart_type == "Regression":
                sns.regplot(x=np.arange(len(y)), y=y, ci=None)
                plt.xticks(np.arange(len(x)), x, rotation=45)

            plt.title(f"{mode.capitalize()} ({chart_type})")
            plt.tight_layout()
            return plt

        except Exception as e:
            print(f"Plot error: {str(e)}")
            return None

    def generate_forecast_plot(self, chart_type="Line"):
        try:
            date_col = self.column_map['date']
            sales_col = self.column_map['sales']

            if not date_col or not sales_col:
                raise ValueError("Date or Sales column not found")

            df = self.data[[date_col, sales_col]].dropna()
            df = df.rename(columns={date_col: "ds", sales_col: "y"})
            df = df.groupby("ds").sum().reset_index()

            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=3, freq='M')
            forecast = model.predict(future)

            forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(6)
            x = np.arange(len(forecast_df))
            y = forecast_df["yhat"].values

            plt.figure(figsize=(10, 6))

            if chart_type == "Line":
                plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", linestyle='--')
                plt.fill_between(forecast_df["ds"],
                                 forecast_df["yhat_lower"],
                                 forecast_df["yhat_upper"],
                                 color='lightblue', alpha=0.3)
            elif chart_type == "Bar":
                plt.bar(forecast_df["ds"], forecast_df["yhat"])
            elif chart_type == "Pie":
                plt.pie(forecast_df["yhat"], labels=forecast_df["ds"].dt.strftime('%b'), autopct='%1.1f%%')
            elif chart_type == "Regression":
                sns.regplot(x=x, y=y, ci=None)
                plt.xticks(x, forecast_df["ds"].dt.strftime('%b'))

            plt.title(f"Forecasted Sales ({chart_type})")
            return plt

        except Exception as e:
            print(f"Forecast error: {str(e)}")
            return None

assistant = BIAssistant()

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"]
).set(
    button_primary_background_fill="#2563eb",
    button_primary_text_color="white",
    body_background_fill="#f4f8fb",
    body_text_color="#1f2937"
)

def refresh_plots(f_type, t_type, p_type):
    return (
        assistant.generate_forecast_plot(f_type),
        assistant.generate_custom_plot(t_type, "trends"),
        assistant.generate_custom_plot(p_type, "product")
    )

with gr.Blocks(theme=custom_theme, title="Business Intelligence Assistant") as demo:
    gr.Markdown("#  💡 INTELLIGENT SALES ANALYTICS AND FORECASTING ENGINE")

    with gr.Tab("Data Setup"):
        with gr.Row():
            data_file = gr.File(label="Upload CSV File", file_types=[".csv"])
            load_btn = gr.Button("Load Data")
        data_status = gr.Textbox(label="Status")
        data_preview = gr.HTML(label="Data Preview")
        init_btn = gr.Button("Initialize System")
        init_status = gr.Textbox(label="Initialization Status")

    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Your business question", placeholder="E.g., What are our top products?")
        ask_btn = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer", lines=5)

    with gr.Tab("Forecasting"):
        forecast_btn = gr.Button("Generate Insights & Forecast")

        forecast_chart_type = gr.Dropdown(
            choices=["Line", "Bar", "Pie", "Regression"],
            value="Line",
            label="Forecast Chart Type"
        )
        forecast_plot = gr.Plot()

        trend_chart_type = gr.Dropdown(
            choices=["Line", "Bar", "Pie", "Regression"],
            value="Line",
            label="Sales Trend Chart Type"
        )
        trend_plot = gr.Plot()

        product_chart_type = gr.Dropdown(
            choices=["Bar", "Pie", "Line", "Regression"],
            value="Bar",
            label="Product Performance Chart Type"
        )
        product_plot = gr.Plot()

    # Event Bindings
    load_btn.click(assistant.load_data, inputs=[data_file], outputs=[data_status, data_preview])
    init_btn.click(assistant.initialize_system, outputs=[init_status])
    ask_btn.click(assistant.ask_question, inputs=[question_input], outputs=[answer_output])

    forecast_btn.click(
        fn=refresh_plots,
        inputs=[forecast_chart_type, trend_chart_type, product_chart_type],
        outputs=[forecast_plot, trend_plot, product_plot]
    )

    forecast_chart_type.change(fn=lambda t: assistant.generate_forecast_plot(t), inputs=forecast_chart_type, outputs=forecast_plot)
    trend_chart_type.change(fn=lambda t: assistant.generate_custom_plot(t, "trends"), inputs=trend_chart_type, outputs=trend_plot)
    product_chart_type.change(fn=lambda t: assistant.generate_custom_plot(t, "product"), inputs=product_chart_type, outputs=product_plot)

if _name_ == "_main_":
    demo.launch()

