"""
generate_report.py
==================
Generates the project report as a PDF.

Usage
-----
    python src/generate_report.py
    # Output: report.pdf
"""

from fpdf import FPDF


class PDF(FPDF):
    """Custom PDF class with consistent header and footer styling."""

    def header(self):
        """Render title bar and rule at the top of every page."""
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "Tabular Regression - Machine Learning Project", align="C")
        self.ln(1)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        """Render the page number at the bottom of every page."""
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def section_title(self, text):
        """Render a bold section heading with a light background fill."""
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(235, 240, 250)
        self.set_x(self.l_margin)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)
        self.set_font("Helvetica", "", 10)

    def subsection_title(self, text):
        """Render a smaller bold subsection heading."""
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_x(self.l_margin)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10)

    def body_text(self, text):
        """Render a paragraph with automatic line wrapping."""
        self.set_font("Helvetica", "", 10)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        """Render a single indented bullet point."""
        self.set_font("Helvetica", "", 10)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, "  - " + text)

    def metric_table(self, headers, rows):
        """
        Render a simple bordered table with alternating row shading.

        Parameters
        ----------
        headers : list of str
        rows : list of list of str
        """
        col_w = 190 // len(headers)
        # Header row
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(210, 220, 240)
        self.set_x(self.l_margin)
        for h in headers:
            self.cell(col_w, 7, h, border=1, fill=True, align="C")
        self.ln()
        # Data rows with alternating shading
        self.set_font("Helvetica", "", 10)
        fill = False
        for row in rows:
            self.set_fill_color(245, 247, 252) if fill else self.set_fill_color(255, 255, 255)
            self.set_x(self.l_margin)
            for cell in row:
                self.cell(col_w, 6.5, str(cell), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(2)


def build_pdf(output_path="report/project_report.pdf"):
    """
    Build and save the full project report PDF.

    Parameters
    ----------
    output_path : str
        Destination file path for the generated PDF.
    """
    pdf = PDF()
    pdf.set_margins(10, 18, 10)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Title block ----
    pdf.set_font("Helvetica", "B", 16)
    pdf.ln(2)
    pdf.cell(0, 10, "Tabular Regression - Machine Learning Project",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Predicting Continuous Targets from High-Dimensional Feature Data",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_draw_color(100, 120, 180)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(6)

    # ================================================================
    # 1. Problem Overview
    # ================================================================
    pdf.section_title("1. Problem Overview")
    pdf.body_text(
        "This project explores regression on a high-dimensional tabular dataset "
        "with 273 numerical input features and two continuous target variables. "
        "The dataset contains 10,000 training samples and 10,000 evaluation samples."
    )
    pdf.body_text(
        "Two prediction tasks are addressed: (1) predicting target01 using a machine "
        "learning ensemble model, and (2) predicting target02 using a lightweight "
        "rule-based model derived from decision tree analysis, suitable for deployment "
        "on constrained devices without any ML runtime."
    )

    # ================================================================
    # 2. Dataset Description
    # ================================================================
    pdf.section_title("2. Dataset Description")
    pdf.body_text(
        "The dataset consists of 273 anonymised numerical features (feat_0 to feat_272). "
        "Features contain a mix of continuous and integer-valued columns. Both training "
        "and evaluation sets contain 10,000 rows."
    )
    pdf.subsection_title("Target Variable Statistics")
    pdf.metric_table(
        ["Target", "Min", "Max", "Mean", "Std"],
        [
            ["target01", "0.096", "1.149", "0.509", "0.226"],
            ["target02", "-1.807", "2.878", "0.288", "0.865"],
        ]
    )
    pdf.body_text(
        "target01 is bounded in a narrow positive range, while target02 spans a broader "
        "range including negative values, suggesting different underlying generating "
        "processes that justify separate modelling strategies."
    )

    # ================================================================
    # 3. Model Development - target01
    # ================================================================
    pdf.section_title("3. Model Development - target01")

    pdf.subsection_title("3.1 Candidate Models")
    pdf.body_text(
        "Four regression approaches were evaluated to identify the most accurate predictor "
        "for target01. A shallow Decision Tree served as an interpretable baseline. Ridge "
        "Regression tested linear separability of the features. Random Forest and Gradient "
        "Boosting were evaluated as ensemble methods known for strong performance on tabular "
        "data [1, 2]."
    )
    pdf.metric_table(
        ["Model", "RMSE", "MAE", "R2"],
        [
            ["Decision Tree (depth 3)", "0.1192", "0.1053", "0.190"],
            ["Ridge Regression",        "0.1197", "0.1055", "0.184"],
            ["Random Forest",           "0.1143", "0.0975", "0.255"],
            ["Gradient Boosting",       "0.0740", "0.0652", "0.688"],
        ]
    )

    pdf.subsection_title("3.2 Gradient Boosting Regressor")
    pdf.body_text(
        "Gradient Boosting [1] builds an additive model in a forward stage-wise manner. "
        "Each tree corrects the residuals of the current ensemble, weighted by the learning "
        "rate. This iterative error-correction mechanism explains the large performance gap "
        "over single-tree and linear baselines on this dataset."
    )
    pdf.body_text("Final hyperparameters (selected by grid search on RMSE):")
    for item in [
        "n_estimators = 200  (number of boosting stages)",
        "max_depth    = 4    (depth of each weak learner)",
        "learning_rate = 0.05  (shrinkage applied per tree contribution)",
        "random_state = 42   (reproducibility)",
    ]:
        pdf.bullet(item)
    pdf.ln(2)

    # ================================================================
    # 4. Rule-Based Model - target02
    # ================================================================
    pdf.section_title("4. Rule-Based Model - target02")

    pdf.subsection_title("4.1 Feature Discovery")
    pdf.body_text(
        "target02 exhibits a structured dependency on a very small subset of features. "
        "A depth-3 DecisionTreeRegressor was fitted on the training data to identify the "
        "most discriminative features and split thresholds. The tree converged to just "
        "three features:"
    )
    for item in [
        "feat_76  (index 76):  primary split - partitions samples at thresholds 0.20 and 0.50",
        "feat_173 (index 173): secondary split for samples where feat_76 <= 0.50",
        "feat_97  (index 97):  secondary split for samples where feat_76 > 0.50",
    ]:
        pdf.bullet(item)
    pdf.ln(2)
    pdf.body_text(
        "A depth-3 tree was chosen because it yields the best trade-off between prediction "
        "quality (RMSE = 0.465, R2 = 0.360) and rule simplicity. Deeper trees add "
        "complexity without meaningful accuracy improvement on the evaluation set."
    )

    pdf.subsection_title("4.2 Extracted Decision Rules")
    pdf.body_text(
        "The tree was translated into explicit threshold conditions. Leaf values represent "
        "the mean training target02 value within each partition, which is the optimal "
        "constant prediction under squared-error loss."
    )
    # Code block
    pdf.set_font("Courier", "", 9)
    pdf.set_fill_color(245, 245, 245)
    for line in [
        "if feat_76 <= 0.20:",
        "    if feat_173 <= 0.48:  target02 =  0.18",
        "    else:                  target02 =  0.55",
        "elif feat_76 <= 0.50:",
        "    if feat_173 <= 0.46:  target02 =  0.82",
        "    else:                  target02 =  1.66",
        "else:  # feat_76 > 0.50",
        "    if   feat_97 <= 0.29: target02 =  0.04",
        "    elif feat_97 <= 0.47: target02 = -0.22",
        "    elif feat_97 <= 0.67: target02 = -0.42",
        "    else:                  target02 = -0.72",
    ]:
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 5.2, line, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)

    # ================================================================
    # 5. Results
    # ================================================================
    pdf.section_title("5. Results")

    pdf.subsection_title("target01 - Gradient Boosting")
    pdf.metric_table(
        ["Metric", "Value"],
        [["RMSE", "0.0740"], ["MAE", "0.0652"], ["R2", "0.688"]]
    )
    pdf.body_text(
        "The Gradient Boosting model achieves R2 = 0.688 on the evaluation set, a 2.7x "
        "improvement in explained variance over the Decision Tree baseline. The 38% RMSE "
        "reduction demonstrates that significant non-linear feature interactions are present "
        "in the data and are successfully captured by the ensemble."
    )

    pdf.subsection_title("target02 - Rule-Based Model")
    pdf.metric_table(
        ["Metric", "Value"],
        [["RMSE", "0.4653"], ["MAE", "0.3419"], ["R2", "0.360"]]
    )
    pdf.body_text(
        "The three-feature, six-leaf rule set explains 36% of the variance in target02 "
        "while remaining fully expressible as simple threshold comparisons. No ML library "
        "is required at inference time, making this model suitable for edge deployment."
    )

    # ================================================================
    # 6. Future Improvements
    # ================================================================
    pdf.section_title("6. Future Improvements")
    for item in [
        "Cross-validated hyperparameter search (RandomizedSearchCV) for target01",
        "Feature importance analysis to remove low-signal features and reduce dimensionality",
        "Explore XGBoost / LightGBM as faster alternatives to sklearn GradientBoosting",
        "Stacking or blending ensemble approaches for target01",
        "Apply SHAP values for model interpretability analysis",
    ]:
        pdf.bullet(item)
    pdf.ln(4)

    # ================================================================
    # References
    # ================================================================
    pdf.section_title("References")
    pdf.set_font("Helvetica", "", 9.5)
    for ref in [
        "[1] Friedman, J. H. (2001). Greedy function approximation: a gradient boosting "
        "machine. Annals of Statistics, 29(5), 1189-1232.",
        "[2] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
        "Journal of Machine Learning Research, 12, 2825-2830.",
    ]:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.5, ref)
        pdf.ln(1)

    pdf.output(output_path)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    import os
    os.makedirs("report", exist_ok=True)
    build_pdf("report/project_report.pdf")
