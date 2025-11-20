import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Payroll Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Payroll Dashboard")

# =========================================================
# 2. CONSTANTS & COLUMN NAMES
# =========================================================
COL_ID = "Tabel"
COL_NAME = "S.A.A"
COL_DEPT = "Departament"
COL_DEPT_GROUP = "Grup departament vlk"
COL_REGION = "Bölgü vlk"
COL_SALARY = "Əmək haqqı"
COL_BONUS_PROJECT = "Birdəfəlik mükafat layihə"
COL_BONUS_TRAINING = "Birdəfəlik mükafat təlim"
COL_BONUS_OTHER = "Birdəfəlik mükafat digər"
COL_DATE = "ay_il"
COL_YEAR = "İl"
COL_MONTH = "Ay"
COL_EOY = "EOY öncəki ilin dekabrına keçirilmiş"
COL_TOTAL = "Total payment"
COL_VACATION = "Məzuniyyət"
COL_BONUS = "Bonuslar"
COL_OTHER2 = "Digər2"
COL_POSITION_GROUP = "Vəzifə  (qruplaşdırılmış)"
COL_BRANCH_CURATOR = "Kurasiya filial"
COL_CURR_SUPER_GROSS = "Cari Super Gross"

MONTH_ORDER = [
    "Yanvar", "Fevral", "Mart", "Aprel", "May", "İyun",
    "İyul", "Avqust", "Sentyabr", "Oktyabr", "Noyabr", "Dekabr"
]

# Breakdown komponentləri – legend AZ, column adları ilə map
COMPONENTS = {
    "Əmək haqqı": COL_SALARY,
    "Bonuslar": COL_BONUS,
    "EOY": COL_EOY,
    "Məzuniyyət": COL_VACATION,
    "Digər ödənişlər": COL_OTHER2,
}
COMP_REVERSE = {v: k for k, v in COMPONENTS.items()}

COMP_COLORS = {
    "Əmək haqqı": "#636EFA",        # blue
    "Bonuslar": "#EF553B",          # red
    "EOY": "#00CC96",               # green
    "Məzuniyyət": "#FFA15A",        # orange
    "Digər ödənişlər": "#AB63FA",   # purple
}

# Group colors for stacked bar charts (tünddən açığa göy)
GROUP_COLORS = {
    "Head Office": "#002855",  # deep navy
    "IT": "#00509e",           # royal blue
    "Branch": "#7fb3ff",       # light blue
}

# Department table payment types (ingiliscə – əvvəlki kimi qalsın)
DEPT_METRICS = {
    "Total payment": COL_TOTAL,
    "Salary": COL_SALARY,
    "Bonuses": COL_BONUS,
    "EOY": COL_EOY,
    "Vacation": COL_VACATION,
    "Project bonus": COL_BONUS_PROJECT,
    "Training bonus": COL_BONUS_TRAINING,
    "Other one-off bonus": COL_BONUS_OTHER,
    "Remaining other": "Remaining other",  # Digər2 - (project+training+other one-off)
}

# Çart selectbox-ları üçün ödəniş növləri (AZ)
PAYMENT_OPTIONS = {
    "Bütün ödənişlər": COL_TOTAL,
    "Əmək haqqı": COL_SALARY,
    "Bonuslar": COL_BONUS,
    "EOY": COL_EOY,
    "Məzuniyyət": COL_VACATION,
    "Layihə bonusu": COL_BONUS_PROJECT,
    "Təlim bonusu": COL_BONUS_TRAINING,
    "Birdəfəlik bonus (digər)": COL_BONUS_OTHER,
    "Qalan digər": "Remaining other",
}

# =========================================================
# 3. DATA LOAD
# =========================================================
uploaded_file = st.sidebar.file_uploader("Upload payroll CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file from the sidebar.")
    st.stop()


@st.cache_data
def load_data(file):
    df_local = pd.read_csv(file, encoding="utf-8-sig")
    return df_local


df = load_data(uploaded_file)

# Numeric məbləğləri təmizləyək
numeric_cols = [
    COL_SALARY,
    COL_EOY,
    COL_TOTAL,
    COL_VACATION,
    COL_BONUS,
    COL_OTHER2,
    COL_BONUS_PROJECT,
    COL_BONUS_TRAINING,
    COL_BONUS_OTHER,
    COL_CURR_SUPER_GROSS,
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Regions və Department group adlarında boşluqları təmizləyək
for col in [COL_REGION, COL_DEPT_GROUP]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Kurasiya filial filter: NA/#N/A → "Other"
df["Kurasiya_filial_filter"] = (
    df.get(COL_BRANCH_CURATOR, pd.Series(index=df.index, dtype="object"))
    .fillna("Other")
    .replace({"#N/A": "Other", "N/A": "Other", "n/a": "Other", "NA": "Other"})
)

# Qalan digər – Digər2-dən project/training/one-off çıxılır
df["Remaining other"] = (
    df[COL_OTHER2]
    - df[COL_BONUS_PROJECT]
    - df[COL_BONUS_TRAINING]
    - df[COL_BONUS_OTHER]
)
df["Remaining other"] = df["Remaining other"].fillna(0)

# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================
def fmt_amount(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.0f}"


def fmt_growth(pct: float) -> str:
    if pd.isna(pct) or not np.isfinite(pct):
        return ""
    return f"{pct:+.1f}%"


def add_growth_by_year(df_year: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df_year = df_year.sort_values(COL_YEAR)
    df_year[f"{value_col}_growth_pct"] = df_year[value_col].pct_change() * 100
    return df_year


def add_growth_per_group(df_long: pd.DataFrame,
                         group_col: str,
                         value_col: str,
                         growth_col: str = "growth_pct") -> pd.DataFrame:
    df_long = df_long.sort_values([group_col, COL_YEAR])
    df_long[growth_col] = (
        df_long.groupby(group_col)[value_col].pct_change() * 100
    )
    return df_long


def drop_zero_component(df_long: pd.DataFrame, component_name: str) -> pd.DataFrame:
    """
    EOY bütün illərdə 0-dırsa, həmin komponenti ümumiyyətlə qrafikdən çıxar.
    """
    if df_long.empty or "Component" not in df_long.columns:
        return df_long
    mask = df_long["Component"] == component_name
    if not mask.any():
        return df_long
    if (df_long.loc[mask, "value"] == 0).all():
        return df_long[~mask]
    return df_long


def compute_breakdowns(df_scope: pd.DataFrame):
    """
    Verilən scope üçün:
      - komponentlər üzrə illik total,
      - həmin komponentdən faydalanan orta işçilər,
      - adam başına düşən məbləğ.
    Məzuniyyət ayrıca komponentdir, Digər ödənişlər Digər2-dir.
    Digər ödənişlər üçün hover-də Layihə/Təlim/Birdəfəlik/Qalan digər breakdown göstərilir.
    """
    if df_scope.empty:
        empty = pd.DataFrame(columns=[COL_YEAR, "Component", "value", "growth_pct"])
        return empty.copy(), empty.copy(), empty.copy()

    # İl üzrə komponent məbləğləri
    year_components_amount = (
        df_scope
        .groupby(COL_YEAR, as_index=False)[
            [COL_SALARY, COL_BONUS, COL_EOY, COL_OTHER2, COL_VACATION]
        ]
        .sum()
    )

    # Digər2 breakdown – layihə, təlim, birdəfəlik, qalan digər
    other_break = (
        df_scope
        .groupby(COL_YEAR, as_index=False)[
            [COL_BONUS_PROJECT, COL_BONUS_TRAINING, COL_BONUS_OTHER]
        ]
        .sum()
        .rename(columns={
            COL_BONUS_PROJECT: "other_proj",
            COL_BONUS_TRAINING: "other_train",
            COL_BONUS_OTHER: "other_once",
        })
    )
    year_components_amount = year_components_amount.merge(other_break, on=COL_YEAR, how="left")
    for col in ["other_proj", "other_train", "other_once"]:
        year_components_amount[col] = year_components_amount[col].fillna(0)

    year_components_amount["other_rest"] = (
        year_components_amount[COL_OTHER2]
        - (year_components_amount["other_proj"]
           + year_components_amount["other_train"]
           + year_components_amount["other_once"])
    )

    # ---- 1) Komponent total-ları (AZN) ----
    comp_total_long = year_components_amount.melt(
        id_vars=[COL_YEAR, "other_proj", "other_train", "other_once", "other_rest"],
        value_vars=list(COMPONENTS.values()),
        var_name="Component_raw",
        value_name="value",
    )
    comp_total_long["Component"] = comp_total_long["Component_raw"].map(COMP_REVERSE)
    comp_total_long = add_growth_per_group(
        comp_total_long,
        group_col="Component",
        value_col="value",
    )
    comp_total_long = drop_zero_component(comp_total_long, "EOY")

    # ---- 2) Komponent üzrə orta əməkdaş sayı ----
    comp_emp_frames = []
    for label, col_name in COMPONENTS.items():
        tmp_col = col_name
        tmp = df_scope[df_scope[tmp_col] > 0]
        if tmp.empty:
            continue
        monthly_emp = (
            tmp.groupby([COL_YEAR, COL_MONTH])[COL_ID]
            .nunique()
            .reset_index(name="employee_count")
        )
        avg_emp = (
            monthly_emp
            .groupby(COL_YEAR, as_index=False)["employee_count"]
            .mean()
        )
        avg_emp["Component"] = label
        avg_emp = avg_emp.rename(columns={"employee_count": "value"})
        comp_emp_frames.append(avg_emp)

    if comp_emp_frames:
        comp_emp_long = pd.concat(comp_emp_frames, ignore_index=True)
        comp_emp_long = add_growth_per_group(
            comp_emp_long,
            group_col="Component",
            value_col="value",
        )
        comp_emp_long = drop_zero_component(comp_emp_long, "EOY")
    else:
        comp_emp_long = pd.DataFrame(columns=[COL_YEAR, "Component", "value", "growth_pct"])

    # ---- 3) Adam başına komponent məbləği ----
    monthly_emp_total = (
        df_scope
        .groupby([COL_YEAR, COL_MONTH])[COL_ID]
        .nunique()
        .reset_index(name="employee_count")
    )
    avg_emp_total = (
        monthly_emp_total
        .groupby(COL_YEAR, as_index=False)["employee_count"]
        .mean()
        .rename(columns={"employee_count": "avg_employees"})
    )

    per_emp_breakdown = year_components_amount[
        [COL_YEAR, COL_SALARY, COL_BONUS, COL_EOY, COL_VACATION, COL_OTHER2,
         "other_proj", "other_train", "other_once", "other_rest"]
    ].merge(
        avg_emp_total, on=COL_YEAR, how="left"
    )

    per_emp_breakdown["avg_employees"] = per_emp_breakdown["avg_employees"].replace(0, np.nan)

    for col_name in COMPONENTS.values():
        per_emp_breakdown[col_name] = (
            per_emp_breakdown[col_name] / per_emp_breakdown["avg_employees"]
        )

    for col in ["other_proj", "other_train", "other_once", "other_rest"]:
        per_emp_breakdown[col] = (
            per_emp_breakdown[col] / per_emp_breakdown["avg_employees"]
        )

    per_emp_long = per_emp_breakdown.melt(
        id_vars=[COL_YEAR, "other_proj", "other_train", "other_once", "other_rest"],
        value_vars=list(COMPONENTS.values()),
        var_name="Component_raw",
        value_name="value",
    )
    per_emp_long["Component"] = per_emp_long["Component_raw"].map(COMP_REVERSE)
    per_emp_long = add_growth_per_group(
        per_emp_long,
        group_col="Component",
        value_col="value",
    )
    per_emp_long = drop_zero_component(per_emp_long, "EOY")

    return comp_total_long, comp_emp_long, per_emp_long


def create_bar_with_growth(df_year: pd.DataFrame,
                           value_col: str,
                           growth_col: str,
                           title: str):
    """Üst 3 bar chart (total, avg employees, per employee)."""
    if df_year.empty:
        return None

    df_plot = df_year.copy()

    fig = px.bar(
        df_plot,
        x=COL_YEAR,
        y=value_col,
        text=df_plot[value_col].apply(fmt_amount),
    )

    max_val = df_plot[value_col].max() if not df_plot.empty else 0
    min_val = min(0, df_plot[value_col].min()) if not df_plot.empty else 0

    fig.update_traces(
        marker_color="#003366",
        textposition="outside",
        textfont=dict(
            size=13,
            color="black",
        ),
        cliponaxis=False,
    )

    fig.update_xaxes(
        title_text="",
        showgrid=False,
        tickmode="array",
        tickvals=df_plot[COL_YEAR],
        ticktext=df_plot[COL_YEAR].astype(str),
    )

    fig.update_yaxes(
        showticklabels=False,
        title_text="",
        showgrid=False,
        range=[min_val, max_val * 1.3 if max_val > 0 else 1],
    )

    fig.update_layout(
        title=dict(
            text=title.replace("\n", "<br>"),
            x=0.0,
            xanchor="left",
        ),
        title_font=dict(size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=10, r=30, t=80, b=80),
    )

    # Growth faiz üçün annotasiya
    for _, row in df_plot.iterrows():
        growth = row.get(growth_col)
        if pd.notna(growth) and np.isfinite(growth):
            fig.add_annotation(
                x=row[COL_YEAR],
                y=row[value_col] / 2,
                text=fmt_growth(growth),
                showarrow=False,
                font=dict(color="white", size=11),
            )
    return fig


def create_breakdown_line(df_long: pd.DataFrame, title: str):
    """
    Breakdown line chart with AZ component labels.
    Hover-da həm məbləğ, həm də artım faizi görünür.
    Digər ödənişlər üçün əlavə olaraq Layihə/Təlim/Birdəfəlik/Qalan digər breakdown göstərilir.
    """
    if df_long.empty:
        return None

    df_plot = df_long.copy()
    df_plot["year_str"] = df_plot[COL_YEAR].astype(str)

    # label: məbləğ + growth
    def make_label(row):
        g = row.get("growth_pct")
        if pd.notna(g):
            return f"{fmt_amount(row['value'])} ({fmt_growth(g)})"
        else:
            return fmt_amount(row["value"])

    df_plot["label"] = df_plot.apply(make_label, axis=1)

    # custom_data: [growth_pct, other_proj, other_train, other_once, other_rest]
    df_plot["growth_pct"] = df_plot.get("growth_pct", np.nan)
    custom_cols = ["growth_pct"]
    for extra_col in ["other_proj", "other_train", "other_once", "other_rest"]:
        if extra_col in df_plot.columns:
            custom_cols.append(extra_col)

    fig = px.line(
        df_plot,
        x="year_str",
        y="value",
        color="Component",
        color_discrete_map=COMP_COLORS,
        markers=True,
        text="label",
        custom_data=custom_cols,
    )

    max_val = df_plot["value"].max() if not df_plot.empty else 0
    min_val = min(0, df_plot["value"].min()) if not df_plot.empty else 0

    fig.update_traces(
        textposition="top center",
        textfont=dict(
            size=12,
            color="black",
        ),
        cliponaxis=False,
    )

    fig.update_yaxes(
        showticklabels=False,
        title_text="",
        showgrid=False,
        range=[min_val, max_val * 1.3 if max_val > 0 else 1],
    )
    fig.update_xaxes(
        title_text="",
        showgrid=False,
        type="category",
    )

    fig.update_layout(
        title=dict(text=title.replace("\n", "<br>"), x=0.0, xanchor="left"),
        title_font=dict(size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="",
        margin=dict(l=10, r=30, t=80, b=80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )

    # hovertemplate – artım faizləri də göstərilsin
    for trace in fig.data:
        name = trace.name
        if name == "Digər ödənişlər" and len(custom_cols) >= 5:
            # customdata[0] = growth_pct
            # customdata[1..4] = other_proj, other_train, other_once, other_rest
            trace.hovertemplate = (
                "İl=%{x}<br>"
                "%{fullData.name}=%{y:,.0f} AZN<br>"
                "Artım=%{customdata[0]:+.1f}%<br>"
                "• Layihə bonusu=%{customdata[1]:,.0f} AZN<br>"
                "• Təlim bonusu=%{customdata[2]:,.0f} AZN<br>"
                "• Digər birdəfəlik bonus=%{customdata[3]:,.0f} AZN<br>"
                "• Qalan digər=%{customdata[4]:,.0f} AZN<br>"
                "<extra></extra>"
            )
        else:
            # Digər komponentlər üçün yalnız məbləğ + artım
            trace.hovertemplate = (
                "İl=%{x}<br>"
                "%{fullData.name}=%{y:,.0f} AZN<br>"
                "Artım=%{customdata[0]:+.1f}%"
                "<extra></extra>"
            )

    return fig


def render_breakdown_row(df_scope: pd.DataFrame,
                         row_title_prefix: str):
    """Verilən scope üçün sağ-sol 3 breakdown line chart çəkir."""
    comp_total_long, comp_emp_long, per_emp_long = compute_breakdowns(df_scope)

    with st.spinner("Loading charts..."):
        c1, c2, c3 = st.columns(3)

        with c1:
            fig = create_breakdown_line(
                comp_total_long,
                title=f"{row_title_prefix} – total payments by component",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data for this chart with current filters.")

        with c2:
            fig = create_breakdown_line(
                comp_emp_long,
                title=f"{row_title_prefix} – employees receiving each component",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data for this chart with current filters.")

        with c3:
            fig = create_breakdown_line(
                per_emp_long,
                title=f"{row_title_prefix} – per-employee amount by component",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data for this chart with current filters.")


def build_department_metric_table(
    df_scope: pd.DataFrame,
    metric_keys: list[str],
    years: list[int],
    mode: str,
) -> pd.DataFrame:
    """
    Department & group table üçün cədvəl:
      - mode="amount"   → ödənilən məbləğ,
      - mode="avg_emp"  → həmin ödənişdən faydalanan orta əməkdaş sayı,
      - mode="per_emp"  → adam başına düşən məbləğ.
    """
    if not metric_keys or not years:
        return pd.DataFrame()

    df_metric = df_scope.copy()
    if df_metric.empty:
        return pd.DataFrame()

    df_metric["Remaining other"] = (
        df_metric[COL_OTHER2]
        - df_metric[COL_BONUS_PROJECT]
        - df_metric[COL_BONUS_TRAINING]
        - df_metric[COL_BONUS_OTHER]
    )

    df_metric = df_metric[df_metric[COL_YEAR].isin(years)]
    if df_metric.empty:
        return pd.DataFrame()

    frames = []

    for metric_key in metric_keys:
        base_col = DEPT_METRICS[metric_key]

        m_df = pd.DataFrame()

        if mode == "amount":
            m_df = (
                df_metric
                .groupby([COL_REGION, COL_DEPT_GROUP, COL_YEAR], as_index=False)[base_col]
                .sum()
                .rename(columns={base_col: "value"})
            )
        else:
            tmp = df_metric[df_metric[base_col] > 0]
            if tmp.empty:
                continue

            monthly_emp = (
                tmp.groupby([COL_REGION, COL_DEPT_GROUP, COL_YEAR, COL_MONTH])[COL_ID]
                .nunique()
                .reset_index(name="emp")
            )
            emp_df = (
                monthly_emp
                .groupby([COL_REGION, COL_DEPT_GROUP, COL_YEAR], as_index=False)["emp"]
                .mean()
                .rename(columns={"emp": "emp_value"})
            )

            if mode == "avg_emp":
                m_df = emp_df.rename(columns={"emp_value": "value"})
            elif mode == "per_emp":
                amount_df = (
                    df_metric
                    .groupby([COL_REGION, COL_DEPT_GROUP, COL_YEAR], as_index=False)[base_col]
                    .sum()
                    .rename(columns={base_col: "amt"})
                )
                m_df = amount_df.merge(
                    emp_df,
                    on=[COL_REGION, COL_DEPT_GROUP, COL_YEAR],
                    how="left",
                )
                m_df["value"] = m_df["amt"] / m_df["emp_value"].replace(0, np.nan)

        if m_df.empty:
            continue

        m_df["Metric"] = metric_key
        frames.append(m_df)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values([COL_REGION, COL_DEPT_GROUP, "Metric", COL_YEAR])
    all_df["growth_pct"] = (
        all_df
        .groupby([COL_REGION, COL_DEPT_GROUP, "Metric"])["value"]
        .pct_change() * 100
    )

    years_sorted = sorted(years)

    unit = {
        "amount": "AZN",
        "avg_emp": "employees",
        "per_emp": "AZN/employee",
    }.get(mode, "AZN")

    rows = []
    for (region, dept), g in all_df.groupby([COL_REGION, COL_DEPT_GROUP]):
        row = {COL_REGION: region, COL_DEPT_GROUP: dept}
        for metric_key in metric_keys:
            gm = g[g["Metric"] == metric_key].sort_values(COL_YEAR)
            for y in years_sorted:
                gy = gm[gm[COL_YEAR] == y]
                if gy.empty:
                    val = np.nan
                    gr = np.nan
                else:
                    r = gy.iloc[0]
                    val = float(r["value"])
                    gr = float(r["growth_pct"]) if pd.notna(r["growth_pct"]) else np.nan

                base_name = f"{y} – {metric_key}"
                row[f"{base_name} ({unit})"] = val
                row[f"{base_name} growth %"] = gr
        rows.append(row)

    return pd.DataFrame(rows)


def build_employee_metric_table(
    df_scope: pd.DataFrame,
    metric_keys: list[str],
    years: list[int],
    mode: str,
) -> pd.DataFrame:
    """
    Employee-level table:
      Group, Department, Employee (S.A.A) səviyyəsində eyni strukturu hesablayır.
    """
    if not metric_keys or not years:
        return pd.DataFrame()

    df_metric = df_scope.copy()
    if df_metric.empty:
        return pd.DataFrame()

    df_metric["Remaining other"] = (
        df_metric[COL_OTHER2]
        - df_metric[COL_BONUS_PROJECT]
        - df_metric[COL_BONUS_TRAINING]
        - df_metric[COL_BONUS_OTHER]
    )

    df_metric = df_metric[df_metric[COL_YEAR].isin(years)]
    if df_metric.empty:
        return pd.DataFrame()

    frames = []

    for metric_key in metric_keys:
        base_col = DEPT_METRICS[metric_key]

        m_df = pd.DataFrame()

        if mode == "amount":
            m_df = (
                df_metric
                .groupby([COL_REGION, COL_DEPT_GROUP, COL_NAME, COL_YEAR], as_index=False)[base_col]
                .sum()
                .rename(columns={base_col: "value"})
            )
        else:
            tmp = df_metric[df_metric[base_col] > 0]
            if tmp.empty:
                continue

            monthly_emp = (
                tmp.groupby([COL_REGION, COL_DEPT_GROUP, COL_NAME, COL_YEAR, COL_MONTH])[COL_ID]
                .nunique()
                .reset_index(name="emp")
            )
            emp_df = (
                monthly_emp
                .groupby([COL_REGION, COL_DEPT_GROUP, COL_NAME, COL_YEAR], as_index=False)["emp"]
                .mean()
                .rename(columns={"emp": "emp_value"})
            )

            if mode == "avg_emp":
                m_df = emp_df.rename(columns={"emp_value": "value"})
            elif mode == "per_emp":
                amount_df = (
                    df_metric
                    .groupby([COL_REGION, COL_DEPT_GROUP, COL_NAME, COL_YEAR], as_index=False)[base_col]
                    .sum()
                    .rename(columns={base_col: "amt"})
                )
                m_df = amount_df.merge(
                    emp_df,
                    on=[COL_REGION, COL_DEPT_GROUP, COL_NAME, COL_YEAR],
                    how="left",
                )
                m_df["value"] = m_df["amt"] / m_df["emp_value"].replace(0, np.nan)

        if m_df.empty:
            continue

        m_df["Metric"] = metric_key
        frames.append(m_df)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values([COL_REGION, COL_DEPT_GROUP, COL_NAME, "Metric", COL_YEAR])
    all_df["growth_pct"] = (
        all_df
        .groupby([COL_REGION, COL_DEPT_GROUP, COL_NAME, "Metric"])["value"]
        .pct_change() * 100
    )

    years_sorted = sorted(years)

    unit = {
        "amount": "AZN",
        "avg_emp": "employees",
        "per_emp": "AZN/employee",
    }.get(mode, "AZN")

    rows = []
    for (region, dept, emp_name), g in all_df.groupby([COL_REGION, COL_DEPT_GROUP, COL_NAME]):
        row = {COL_REGION: region, COL_DEPT_GROUP: dept, COL_NAME: emp_name}
        for metric_key in metric_keys:
            gm = g[g["Metric"] == metric_key].sort_values(COL_YEAR)
            for y in years_sorted:
                gy = gm[gm[COL_YEAR] == y]
                if gy.empty:
                    val = np.nan
                    gr = np.nan
                else:
                    r = gy.iloc[0]
                    val = float(r["value"])
                    gr = float(r["growth_pct"]) if pd.notna(r["growth_pct"]) else np.nan

                base_name = f"{y} – {metric_key}"
                row[f"{base_name} ({unit})"] = val
                row[f"{base_name} growth %"] = gr
        rows.append(row)

    return pd.DataFrame(rows)


def compute_group_year_metrics(df_scope: pd.DataFrame, payment_col: str) -> pd.DataFrame:
    """
    Group (Head Office / IT / Branch) üzrə:
      - total (seçilən payment type),
      - avg_employees (həmin paymenti alan orta işçilər),
      - per_employee (payment / avg_employees).
    """
    if df_scope.empty:
        return pd.DataFrame()

    dfm = df_scope.copy()
    if payment_col == "Remaining other":
        dfm["metric_value"] = dfm["Remaining other"]
    else:
        dfm["metric_value"] = dfm[payment_col]

    totals = (
        dfm
        .groupby([COL_REGION, COL_YEAR], as_index=False)["metric_value"]
        .sum()
        .rename(columns={"metric_value": "total_payment"})
    )

    monthly_emp_group = (
        dfm[dfm["metric_value"] > 0]
        .groupby([COL_REGION, COL_YEAR, COL_MONTH])[COL_ID]
        .nunique()
        .reset_index(name="employee_count")
    )

    year_emp_group = (
        monthly_emp_group
        .groupby([COL_REGION, COL_YEAR], as_index=False)["employee_count"]
        .mean()
        .rename(columns={"employee_count": "avg_employees"})
    )

    metrics = totals.merge(year_emp_group, on=[COL_REGION, COL_YEAR], how="left")
    metrics["per_employee"] = metrics["total_payment"] / metrics["avg_employees"]

    metrics = metrics.sort_values([COL_REGION, COL_YEAR])

    for col in ["total_payment", "avg_employees", "per_employee"]:
        metrics[f"{col}_growth_pct"] = (
            metrics
            .groupby(COL_REGION)[col]
            .pct_change() * 100
        )

    return metrics


def create_clustered_bar_group(metrics_group: pd.DataFrame,
                               value_col: str,
                               growth_col: str,
                               title: str):
    """
    Stacked bar: hər il üçün Head Office / IT / Branch üst-üstə yığılır.
    Hər stack-in içində məbləğ + artım faizi, faiz yeni sətirdə.
    Legend aşağıda, yazılar horizontal və eyni ölçüdədir.
    """
    if metrics_group.empty:
        return None

    df_plot = metrics_group.copy()

    def make_label(row):
        g = row.get(growth_col)
        if pd.notna(g) and np.isfinite(g):
            # Məbləğ və faiz iki sətirdə
            return f"{fmt_amount(row[value_col])}<br>({fmt_growth(g)})"
        else:
            return fmt_amount(row[value_col])

    df_plot["label"] = df_plot.apply(make_label, axis=1)

    fig = px.bar(
        df_plot,
        x=COL_YEAR,
        y=value_col,
        color=COL_REGION,
        barmode="relative",            # stacked
        text="label",
        color_discrete_map=GROUP_COLORS,
    )

    yearly_sum = df_plot.groupby(COL_YEAR)[value_col].sum()
    max_val = yearly_sum.max() if not yearly_sum.empty else 0
    min_val = min(0, yearly_sum.min()) if not yearly_sum.empty else 0

    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=11, color="white"),
        textangle=0,
        cliponaxis=False,
    )

    fig.update_layout(
        uniformtext_minsize=11,
        uniformtext_mode="show",
    )

    years_sorted = sorted(df_plot[COL_YEAR].unique())
    fig.update_xaxes(
        title_text="",
        showgrid=False,
        tickmode="array",
        tickvals=years_sorted,
        ticktext=[str(int(x)) for x in years_sorted],
    )
    fig.update_yaxes(
        title_text="",
        showgrid=False,
        showticklabels=False,
        range=[min_val, max_val * 1.1 if max_val > 0 else 1],
    )

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        title_font=dict(size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Group",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=10, r=30, t=80, b=80),
    )

    return fig


def create_forecast_chart(df_scope: pd.DataFrame, use_first9: bool):
    """
    Forecast total payment:
      - use_first9=True  → first 9 months only, forecast 2026 & 2027 (first 9 months)
      - use_first9=False → full-year totals, 2025 Q4 estimated using historical Q4/first9 ratio,
                           forecast 2026 & 2027 (full year)

    Label-lərdə həm məbləğ, həm də əvvəlki ilə görə artım faizi göstərilir.
    """
    if df_scope.empty:
        return None

    d = df_scope.copy()
    d = d[pd.notna(d[COL_YEAR])]
    if d.empty:
        return None

    monthly = (
        d.groupby([COL_YEAR, COL_MONTH], as_index=False)[COL_TOTAL]
        .sum()
        .rename(columns={COL_TOTAL: "total_payment"})
    )
    if monthly.empty:
        return None

    years = sorted(monthly[COL_YEAR].unique())
    if len(years) < 2:
        return None

    if use_first9:
        rows = []
        first9_months = set(MONTH_ORDER[:9])
        for y in years:
            df_y = monthly[monthly[COL_YEAR] == y]
            first9_total = df_y[df_y[COL_MONTH].isin(first9_months)]["total_payment"].sum()
            if first9_total > 0:
                rows.append((y, first9_total))
        year_df = pd.DataFrame(rows, columns=[COL_YEAR, "total_payment"])
        if year_df.empty or year_df[COL_YEAR].nunique() < 2:
            return None

        x = year_df[COL_YEAR].astype(float).values
        y_vals = year_df["total_payment"].values
        a, b = np.polyfit(x, y_vals, 1)

        max_year = int(year_df[COL_YEAR].max())
        forecast_years = np.array([max_year + 1, max_year + 2], dtype=float)
        forecast_vals = a * forecast_years + b

        hist = year_df.copy()
        hist["Tip"] = "History"

        fut = pd.DataFrame({
            COL_YEAR: forecast_years.astype(int),
            "total_payment": forecast_vals,
            "Tip": "Forecast",
        })

        combined = pd.concat([hist, fut], ignore_index=True)
        title = "Total payment forecast (first 9 months, 2026–2027)"
    else:
        rows = []
        first9_months = set(MONTH_ORDER[:9])
        q4_months = set(MONTH_ORDER[9:])  # Okt–Dek

        ratios = []
        for y in years:
            df_y = monthly[monthly[COL_YEAR] == y]
            first9_total = df_y[df_y[COL_MONTH].isin(first9_months)]["total_payment"].sum()
            q4_total = df_y[df_y[COL_MONTH].isin(q4_months)]["total_payment"].sum()
            if first9_total > 0 and q4_total > 0 and y < 2025:
                ratios.append(q4_total / first9_total)

        q4_factor = np.mean(ratios) if ratios else 3 / 9

        for y in years:
            df_y = monthly[monthly[COL_YEAR] == y]
            if y == 2025:
                first9_total = df_y[df_y[COL_MONTH].isin(first9_months)]["total_payment"].sum()
                est_q4 = first9_total * q4_factor
                annual_total = first9_total + est_q4
            else:
                annual_total = df_y["total_payment"].sum()
            if annual_total > 0:
                rows.append((y, annual_total))

        year_df = pd.DataFrame(rows, columns=[COL_YEAR, "total_payment"])
        if year_df.empty or year_df[COL_YEAR].nunique() < 2:
            return None

        x = year_df[COL_YEAR].astype(float).values
        y_vals = year_df["total_payment"].values
        a, b = np.polyfit(x, y_vals, 1)

        max_year = int(year_df[COL_YEAR].max())
        forecast_years = np.array([max_year + 1, max_year + 2], dtype=float)
        forecast_vals = a * forecast_years + b

        hist = year_df.copy()
        hist["Tip"] = "History"

        fut = pd.DataFrame({
            COL_YEAR: forecast_years.astype(int),
            "total_payment": forecast_vals,
            "Tip": "Forecast",
        })

        combined = pd.concat([hist, fut], ignore_index=True)
        title = "Total payment forecast (full year, 2026–2027)"

    # Artım faizləri
    combined = combined.sort_values(COL_YEAR)
    combined["growth_pct"] = combined["total_payment"].pct_change() * 100

    def make_label(row):
        g = row.get("growth_pct")
        if pd.notna(g) and np.isfinite(g):
            return f"{fmt_amount(row['total_payment'])} ({fmt_growth(g)})"
        else:
            return fmt_amount(row["total_payment"])

    combined["label"] = combined.apply(make_label, axis=1)

    fig = px.line(
        combined,
        x=COL_YEAR,
        y="total_payment",
        color="Tip",
        markers=True,
        text="label",
        category_orders={COL_YEAR: sorted(combined[COL_YEAR].unique())},
    )
    max_val = combined["total_payment"].max()
    min_val = min(0, combined["total_payment"].min())

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=12, color="black"),
        cliponaxis=False,
    )

    for trace in fig.data:
        if trace.name == "Forecast":
            trace.line = dict(dash="dash", width=3)
        else:
            trace.line = dict(width=3)

    fig.update_xaxes(
        title_text="Year",
        showgrid=False,
        tickmode="array",
        tickvals=sorted(combined[COL_YEAR].unique()),
        ticktext=[str(int(v)) for v in sorted(combined[COL_YEAR].unique())],
    )
    fig.update_yaxes(
        title_text="",
        showgrid=False,
        showticklabels=False,
        range=[min_val, max_val * 1.3 if max_val > 0 else 1],
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.0,
            xanchor="left",
        ),
        title_font=dict(size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="",
        margin=dict(l=10, r=30, t=80, b=80),
    )

    return fig


# =========================================================
# 5. FILTERS (GLOBAL FOR TOP CHARTS)
# =========================================================
st.sidebar.header("Filters (top charts)")

all_years = sorted(df[COL_YEAR].dropna().unique())
selected_years = st.sidebar.multiselect(
    "Years",
    options=all_years,
    default=all_years,
)

all_months = [m for m in MONTH_ORDER if m in df[COL_MONTH].dropna().unique()]
selected_months = st.sidebar.multiselect(
    "Months",
    options=all_months,
    default=all_months,
)

limit_months = st.sidebar.checkbox("Limit by first N months from January")
month_limit_n = None
if limit_months:
    max_months = len(MONTH_ORDER)
    month_limit_n = st.sidebar.number_input(
        "Number of months (from January)",
        min_value=1,
        max_value=max_months,
        value=min(9, max_months),
        step=1,
    )

all_regions = sorted(df[COL_REGION].dropna().unique())
selected_regions = st.sidebar.multiselect(
    "Group (Head Office / IT / Branch)",
    options=all_regions,
    default=all_regions,
)

all_depts = sorted(df[COL_DEPT_GROUP].dropna().unique())
selected_depts = st.sidebar.multiselect(
    "Department group",
    options=all_depts,
    default=all_depts,
)

all_curators = sorted(df["Kurasiya_filial_filter"].dropna().unique())
selected_curators = st.sidebar.multiselect(
    "Branch curator",
    options=all_curators,
    default=all_curators,
)

all_positions = sorted(df[COL_POSITION_GROUP].dropna().unique())
selected_positions = st.sidebar.multiselect(
    "Position group",
    options=all_positions,
    default=all_positions,
)

# İl / ay / departament / kurasiya / vəzifə filtrləri
base_filtered = df.copy()
if selected_years:
    base_filtered = base_filtered[base_filtered[COL_YEAR].isin(selected_years)]

if limit_months and month_limit_n is not None:
    allowed_months = set(MONTH_ORDER[: int(month_limit_n)])
    month_filter_list = [m for m in all_months if m in allowed_months]
else:
    month_filter_list = selected_months

if month_filter_list:
    base_filtered = base_filtered[base_filtered[COL_MONTH].isin(month_filter_list)]

if selected_depts:
    base_filtered = base_filtered[base_filtered[COL_DEPT_GROUP].isin(selected_depts)]

if selected_curators:
    base_filtered = base_filtered[base_filtered["Kurasiya_filial_filter"].isin(selected_curators)]

if selected_positions:
    base_filtered = base_filtered[base_filtered[COL_POSITION_GROUP].isin(selected_positions)]

# Group filter yalnız bəzi chartlara təsir edir
filtered = base_filtered.copy()
if selected_regions:
    filtered = filtered[filtered[COL_REGION].isin(selected_regions)]

if filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

# =========================================================
# 6. KEY METRICS BY YEAR (SEÇİLƏN PAYMENT TYPE ÜZRƏ)
# =========================================================
st.subheader("Key metrics by year")

key_payment_choice = st.selectbox(
    "Payment type (for key metrics charts)",
    options=list(PAYMENT_OPTIONS.keys()),
    index=0,
)

key_col = PAYMENT_OPTIONS[key_payment_choice]

data_for_key = filtered.copy()
if key_col == "Remaining other":
    data_for_key["metric_value"] = data_for_key["Remaining other"]
else:
    data_for_key["metric_value"] = data_for_key[key_col]

year_total = (
    data_for_key
    .groupby(COL_YEAR, as_index=False)["metric_value"]
    .sum()
    .rename(columns={"metric_value": "total_payment"})
)

monthly_emp = (
    data_for_key[data_for_key["metric_value"] > 0]
    .groupby([COL_YEAR, COL_MONTH])[COL_ID]
    .nunique()
    .reset_index(name="employee_count")
)

year_emp_avg = (
    monthly_emp
    .groupby(COL_YEAR, as_index=False)["employee_count"]
    .mean()
    .rename(columns={"employee_count": "avg_employees"})
)

metrics = pd.merge(year_total, year_emp_avg, on=COL_YEAR, how="left")
metrics["per_employee"] = metrics["total_payment"] / metrics["avg_employees"]

metrics = add_growth_by_year(metrics, "total_payment")
metrics = add_growth_by_year(metrics, "avg_employees")
metrics = add_growth_by_year(metrics, "per_employee")

with st.spinner("Loading charts..."):
    c1, c2, c3 = st.columns(3)

    with c1:
        fig1 = create_bar_with_growth(
            metrics,
            value_col="total_payment",
            growth_col="total_payment_growth_pct",
            title="Total amount by year (selected payment type)",
        )
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No data for this chart with current filters.")

    with c2:
        fig2 = create_bar_with_growth(
            metrics,
            value_col="avg_employees",
            growth_col="avg_employees_growth_pct",
            title="Average employees with selected payment by year",
        )
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for this chart with current filters.")

    with c3:
        fig3 = create_bar_with_growth(
            metrics,
            value_col="per_employee",
            growth_col="per_employee_growth_pct",
            title="Per-employee amount by year (selected payment type)",
        )
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No data for this chart with current filters.")

st.markdown(
    "_Qeyd: Bu blokda yuxarıdakı selectbox-da seçilən ödəniş növü üzrə illərə görə üç əsas göstərici təqdim olunur: "
    "illik ümumi məbləğ, həmin ödənişi alan orta aylıq əməkdaş sayı və bir əməkdaşa düşən orta məbləğ. "
    "Hər sütunun içində əvvəlki ilə nisbətən artım/faiz dəyişikliyi ayrıca qeyd olunub. İl, ay, departament, "
    "filial kurasiyası və vəzifə filtrinə uyğun olaraq bütün hesablamalar dinamik yenilənir._"
)

# =========================================================
# 7. BREAKDOWN BY COMPONENTS (ALL GROUPS)
# =========================================================
st.subheader("Breakdown by components (all groups)")

render_breakdown_row(filtered, "Ümumi")

st.markdown(
    "_Qeyd: Bu sətirdə seçdiyiniz filtrə uyğun olaraq bütün qruplar üzrə (Head Office, IT, Branch birlikdə) "
    "komponentlərin illik strukturu göstərilir. Birinci qrafikdə hər komponent üzrə ümumi məbləğ, ikinci qrafikdə "
    "həmin komponentdən aylıq faydalanan əməkdaşların orta sayı, üçüncü qrafikdə isə bir əməkdaşa düşən orta məbləğ "
    "əks olunur. Məzuniyyət ayrıca xətt kimi göstərilir, Digər ödənişlər xəttinin hover hissəsində isə Layihə, Təlim, "
    "Birdəfəlik bonus və Qalan digər hissələri ayrıca breakdown şəklində görünür. Hər nöqtənin yanında və hover-də "
    "məbəğlə yanaşı, əvvəlki ilə nisbətən artım/faiz dəyişikliyi də qeyd olunur. EOY yalnız faktiki ödəniş olduqda "
    "qrafikə daxil edilir; seçilmiş periodda EOY 0-dırsa, əlavə sətir tutmaması üçün göstərilmir._"
)

# =========================================================
# 8. GROUP COMPARISON – STACKED (SEÇİLƏN PAYMENT TYPE ÜZRƏ)
# =========================================================
st.subheader("Group comparison – total, employees, per-employee")

group_payment_choice = st.selectbox(
    "Payment type (for group comparison charts)",
    options=list(PAYMENT_OPTIONS.keys()),
    index=0,
)

group_payment_col = PAYMENT_OPTIONS[group_payment_choice]

group_metrics = compute_group_year_metrics(base_filtered, group_payment_col)

with st.spinner("Loading charts..."):
    c1, c2, c3 = st.columns(3)

    with c1:
        fig_g1 = create_clustered_bar_group(
            group_metrics,
            value_col="total_payment",
            growth_col="total_payment_growth_pct",
            title="Total amount by group & year (selected payment type)",
        )
        if fig_g1:
            st.plotly_chart(fig_g1, use_container_width=True)
        else:
            st.info("No data for this chart with current filters.")

    with c2:
        fig_g2 = create_clustered_bar_group(
            group_metrics,
            value_col="avg_employees",
            growth_col="avg_employees_growth_pct",
            title="Average employees with selected payment by group & year",
        )
        if fig_g2:
            st.plotly_chart(fig_g2, use_container_width=True)
        else:
            st.info("No data for this chart with current filters.")

    with c3:
        fig_g3 = create_clustered_bar_group(
            group_metrics,
            value_col="per_employee",
            growth_col="per_employee_growth_pct",
            title="Per-employee amount by group & year (selected payment type)",
        )
        if fig_g3:
            st.plotly_chart(fig_g3, use_container_width=True)
        else:
            st.info("No data for this chart with current filters.")

st.markdown(
    "_Qeyd: Bu blok Head Office, IT və Branch qrupları üzrə seçdiyiniz ödəniş növünün illər boyu dinamikasını "
    "müqayisə edir. Hər sətirdə müvafiq olaraq ümumi məbləğ, həmin ödənişi alan orta aylıq əməkdaş sayı və "
    "adam başına düşən orta məbləğ stacked sütunlar formasında göstərilir. Sütunlar tünddən açığa doğru göy çalarları ilə "
    "qruplara bölünür, hər stack-in içində isə məbləğ və əvvəlki ilə nisbətən artım/faiz dəyişikliyi iki sətirdə oxunaqlı "
    "şəkildə qeyd olunub. Bu blokda Group filteri istifadə olunmur, çünki məqsəd üç qrupun bir-biri ilə "
    "tam müqayisəsini qorumaqdır; digər filterlər (il, ay, departament və s.) isə tətbiq edilir._"
)

# =========================================================
# 9. BREAKDOWN BY COMPONENTS – HEAD OFFICE / IT / BRANCH
# =========================================================
st.subheader("Breakdown by components – Head Office / IT / Branch")

regions = ["Head Office", "IT", "Branch"]
region_breakdowns = {}

for region in regions:
    df_region = base_filtered[base_filtered[COL_REGION] == region]
    comp_total_long, comp_emp_long, per_emp_long = compute_breakdowns(df_region)
    region_breakdowns[region] = {
        "total": comp_total_long,
        "emp": comp_emp_long,
        "peremp": per_emp_long,
    }

# Row 1 – total payments
with st.spinner("Loading charts..."):
    c1, c2, c3 = st.columns(3)
    for col, region in zip([c1, c2, c3], regions):
        with col:
            df_long = region_breakdowns[region]["total"]
            if df_long.empty:
                st.info(f"No data for {region} with current filters.")
            else:
                fig = create_breakdown_line(
                    df_long,
                    title=f"{region} – total payments by component",
                )
                st.plotly_chart(fig, use_container_width=True)

# Row 2 – employees
with st.spinner("Loading charts..."):
    c1, c2, c3 = st.columns(3)
    for col, region in zip([c1, c2, c3], regions):
        with col:
            df_long = region_breakdowns[region]["emp"]
            if df_long.empty:
                st.info(f"No data for {region} with current filters.")
            else:
                fig = create_breakdown_line(
                    df_long,
                    title=f"{region} – employees receiving components",
                )
                st.plotly_chart(fig, use_container_width=True)

# Row 3 – per-employee
with st.spinner("Loading charts..."):
    c1, c2, c3 = st.columns(3)
    for col, region in zip([c1, c2, c3], regions):
        with col:
            df_long = region_breakdowns[region]["peremp"]
            if df_long.empty:
                st.info(f"No data for {region} with current filters.")
            else:
                fig = create_breakdown_line(
                    df_long,
                    title=f"{region} – per-employee amount by component",
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "_Qeyd: Bu blokda hər bir group (Head Office, IT, Branch) üçün komponentlər üzrə illik struktur ayrıca göstərilir. "
    "Sətirlər ardıcıllıqla (1) ümumi məbləğ, (2) həmin komponentdən faydalanan əməkdaş sayı, (3) adam başına düşən "
    "məbləğ üzrə qurulub və üç group yan-yana yerləşdirilib ki, vizual müqayisə daha rahat olsun. Məzuniyyət burada da "
    "ayrı xətt kimi görünür, Digər ödənişlər xəttində hover zamanı Layihə/Təlim/Birdəfəlik/Qalan digər hissələri "
    "detallı şəkildə göstərilir. Hər nöqtə üçün həm məbləğ, həm də illik artım/faiz dəyişikliyi qeyd olunur._"
)

# =========================================================
# 10. DEPARTMENT & GROUP TABLE
# =========================================================
st.subheader("Department & group table")

table_metric_keys = st.multiselect(
    "Payment type(s) (for department table)",
    options=list(DEPT_METRICS.keys()),
    default=list(DEPT_METRICS.keys()),
)

table_measure = st.selectbox(
    "Measure type (for department table)",
    options=[
        "Total amount paid",
        "Average employees paid",
        "Amount per employee",
    ],
)

mode_map = {
    "Total amount paid": "amount",
    "Average employees paid": "avg_emp",
    "Amount per employee": "per_emp",
}
table_mode = mode_map[table_measure]

table_years_all = sorted(df[COL_YEAR].dropna().unique())
table_selected_years = st.multiselect(
    "Years (for department table)",
    options=table_years_all,
    default=table_years_all,
)

table_regions_all = sorted(df[COL_REGION].dropna().unique())
table_selected_regions = st.multiselect(
    "Group (for department table)",
    options=table_regions_all,
    default=table_regions_all,
)

table_months_all = [m for m in MONTH_ORDER if m in df[COL_MONTH].dropna().unique()]
table_selected_months = st.multiselect(
    "Months (for department table)",
    options=table_months_all,
    default=table_months_all,
)

df_table_base = df.copy()
if table_selected_regions:
    df_table_base = df_table_base[df_table_base[COL_REGION].isin(table_selected_regions)]

if table_selected_months:
    df_table_base = df_table_base[df_table_base[COL_MONTH].isin(table_selected_months)]

dept_table = build_department_metric_table(df_table_base, table_metric_keys, table_selected_years, table_mode)

if dept_table.empty:
    st.info("No data for the selected department table filters.")
else:
    # Region -> Group, Group Departament -> Department
    dept_table_display = dept_table.rename(columns={COL_REGION: "Group", COL_DEPT_GROUP: "Department"})

    amount_cols = [c for c in dept_table_display.columns if "(" in c and ")" in c and "growth %" not in c]
    growth_cols = [c for c in dept_table_display.columns if "growth %" in c]

    def fmt_growth_cell(x):
        if pd.isna(x):
            return ""
        return f"{x:+.1f}%"

    fmt_dict = {}
    for col in amount_cols:
        fmt_dict[col] = fmt_amount
    for col in growth_cols:
        fmt_dict[col] = fmt_growth_cell

    def growth_color(val):
        if pd.isna(val):
            return ""
        try:
            v = float(val)
        except Exception:
            return ""
        if v < 0:
            return "color: darkgreen; font-weight: 600;"
        if v > 0:
            return "color: darkred; font-weight: 600;"
        return "color: black;"

    styled = (
        dept_table_display
        .style
        .format(fmt_dict)
        .applymap(growth_color, subset=growth_cols)
    )

    with st.spinner("Loading department table..."):
        st.dataframe(styled, use_container_width=True)

st.markdown(
    "_Qeyd: Bu cədvəldə Group (Head Office / IT / Branch) və Department səviyyəsində seçdiyiniz ödəniş növləri üçün "
    "illər üzrə ya ümumi məbləğ, ya həmin ödənişi alan orta aylıq əməkdaş sayı, ya da bir əməkdaşa düşən orta məbləğ "
    "göstərilir. Hər il üçün uyğun artım/faiz dəyişikliyi ayrıca sütunda verilib; mənfi artımlar tünd yaşıl, müsbət "
    "artımlar isə tünd qırmızı rənglə vurğulanır. Cədvəli istənilən sütuna görə sort edərək yüksək artım və ya azalma "
    "olan departamentləri asanlıqla tapmaq mümkündür; yuxarıdakı filterlər yalnız bu blok üçün tətbiq olunur._"
)

# =========================================================
# 11. EMPLOYEE-LEVEL TABLE
# =========================================================
st.subheader("Employee-level table")

# Eyni df_table_base, metric_keys, years, mode istifadə olunur
emp_dept_options = sorted(df_table_base[COL_DEPT_GROUP].dropna().unique())
emp_selected_depts = st.multiselect(
    "Departments (for employee table)",
    options=emp_dept_options,
    default=emp_dept_options,
)

df_emp_scope = df_table_base.copy()
if emp_selected_depts:
    df_emp_scope = df_emp_scope[df_emp_scope[COL_DEPT_GROUP].isin(emp_selected_depts)]

emp_table = build_employee_metric_table(df_emp_scope, table_metric_keys, table_selected_years, table_mode)

if emp_table.empty:
    st.info("No data for the selected employee-level table filters.")
else:
    emp_display = emp_table.rename(
        columns={
            COL_REGION: "Group",
            COL_DEPT_GROUP: "Department",
            COL_NAME: "Employee",
        }
    )

    amount_cols_emp = [c for c in emp_display.columns if "(" in c and ")" in c and "growth %" not in c]
    growth_cols_emp = [c for c in emp_display.columns if "growth %" in c]

    def fmt_growth_cell_emp(x):
        if pd.isna(x):
            return ""
        return f"{x:+.1f}%"

    fmt_dict_emp = {}
    for col in amount_cols_emp:
        fmt_dict_emp[col] = fmt_amount
    for col in growth_cols_emp:
        fmt_dict_emp[col] = fmt_growth_cell_emp

    def growth_color_emp(val):
        if pd.isna(val):
            return ""
        try:
            v = float(val)
        except Exception:
            return ""
        if v < 0:
            return "color: darkgreen; font-weight: 600;"
        if v > 0:
            return "color: darkred; font-weight: 600;"
        return "color: black;"

    styled_emp = (
        emp_display
        .style
        .format(fmt_dict_emp)
        .applymap(growth_color_emp, subset=growth_cols_emp)
    )

    with st.spinner("Loading employee table..."):
        st.dataframe(styled_emp, use_container_width=True)

st.markdown(
    "_Qeyd: Bu cədvəl daha dərin analiz üçün eyni strukturu əməkdaş səviyyəsinə qədər endirir. Hər sətrdə Group, "
    "Department və Employee (S.A.A) göstərilir və yuxarıdakı payment növləri və ölçü seçimlərinə uyğun olaraq "
    "illər üzrə ödəniş məbləği, həmin ödənişi alan orta aylıq əməkdaş sayı və ya adam başına düşən məbləğ əks olunur. "
    "Departament filteri yalnız bu cədvələ aiddir; bununla, misal üçün, konkret bir departament daxilində əməkdaşları "
    "illər üzrə artım faizlərinə görə sort edərək ən çox artan və ya azalan şəxsləri müəyyənləşdirmək mümkündür._"
)

# =========================================================
# 12. MONTHLY DYNAMICS BY PAYMENT TYPE
# =========================================================
st.subheader("Monthly dynamics by payment type")

monthly_payment_choice = st.selectbox(
    "Payment type (for monthly dynamics chart)",
    options=list(PAYMENT_OPTIONS.keys()),
    index=0,
)

monthly_col = PAYMENT_OPTIONS[monthly_payment_choice]

data_monthly = filtered.copy()
if monthly_col == "Remaining other":
    data_monthly["metric_value"] = data_monthly["Remaining other"]
else:
    data_monthly["metric_value"] = data_monthly[monthly_col]

monthly_series = (
    data_monthly
    .groupby([COL_YEAR, COL_MONTH], as_index=False)["metric_value"]
    .sum()
)

# Month order üçün category
monthly_series[COL_MONTH] = pd.Categorical(
    monthly_series[COL_MONTH],
    categories=MONTH_ORDER,
    ordered=True,
)

with st.spinner("Loading monthly dynamics..."):
    if monthly_series.empty:
        st.info("No data for the selected monthly dynamics filters.")
    else:
        fig_m = px.line(
            monthly_series.sort_values([COL_YEAR, COL_MONTH]),
            x=COL_MONTH,
            y="metric_value",
            color=COL_YEAR,
            markers=True,
            category_orders={COL_MONTH: MONTH_ORDER},
        )
        fig_m.update_traces(
            hovertemplate=(
                "İl=%{fullData.name}<br>"
                "Ay=%{x}<br>"
                "Məbləğ=%{y:,.0f} AZN"
                "<extra></extra>"
            ),
            line=dict(width=3),
        )
        max_val = monthly_series["metric_value"].max()
        min_val = min(0, monthly_series["metric_value"].min())
        fig_m.update_yaxes(
            title_text="",
            showgrid=False,
            showticklabels=False,
            range=[min_val, max_val * 1.3 if max_val > 0 else 1],
        )
        fig_m.update_xaxes(
            title_text="Month",
            showgrid=False,
        )
        fig_m.update_layout(
            title=dict(
                text="Monthly dynamics across selected years (by payment type)",
                x=0.0,
                xanchor="left",
            ),
            title_font=dict(size=14),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend_title_text="Year",
            margin=dict(l=10, r=30, t=80, b=80),
        )
        st.plotly_chart(fig_m, use_container_width=True)

st.markdown(
    "_Qeyd: Bu qrafik seçdiyiniz payment növü üzrə seçilmiş illər və ümumi filterlər (group, departament, filial, "
    "vəzifə və s.) çərçivəsində aylıq dinamikaları göstərir. X oxunda aylar (Yanvar–Dekabr), rənglərlə isə illər "
    "fərqləndirilir. Bu sayədə, məsələn, 2023 və 2024-cü illərin eyni aylarında ödəniş səviyyələrini birbaşa "
    "müqayisə etmək və mövsümi dəyişiklikləri izləmək olur._"
)

# =========================================================
# 13. FORECAST – BOTTOM
# =========================================================
st.subheader("Forecast – total payments")

forecast_use_first9 = st.checkbox(
    "Use only first 9 months for forecast (do not estimate Q4 for 2025)",
    value=False,
)

# Forecast üçün year filter-dən asılı olmayan scope (amma digər filterlər tətbiq olunur)
forecast_filtered = df.copy()
if selected_regions:
    forecast_filtered = forecast_filtered[forecast_filtered[COL_REGION].isin(selected_regions)]
if selected_depts:
    forecast_filtered = forecast_filtered[forecast_filtered[COL_DEPT_GROUP].isin(selected_depts)]
if selected_curators:
    forecast_filtered = forecast_filtered[forecast_filtered["Kurasiya_filial_filter"].isin(selected_curators)]
if selected_positions:
    forecast_filtered = forecast_filtered[forecast_filtered[COL_POSITION_GROUP].isin(selected_positions)]

with st.spinner("Loading forecast..."):
    fig_fc = create_forecast_chart(forecast_filtered, forecast_use_first9)
    if fig_fc:
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info("Not enough historical data to build a forecast.")

st.markdown(
    "_Qeyd: Forecast qrafikində Total payment üzrə tarixi trendə əsaslanaraq gələcək illər üçün proqnoz qurulur. "
    "Əgər aşağıdakı checkbox aktivdirsə, hər il üçün yalnız ilk 9 ayın (Yanvar–Sentyabr) cəmi nəzərə alınır və "
    "2026–2027-ci illər üçün eyni tempdə artım davam edəcəyi fərz olunur. Checkbox söndürülüdürsə, 2025-ci ilin son "
    "3 ayı əvvəlki illərin Q4/ilk 9 ay nisbətlərinə əsasən təxmini hesablanır və 2026–2027-ci illər üzrə tam illik "
    "proqnoz göstərilir. Qrafikdə tarixi və proqnoz xətləri fərqli cizgi üslubu ilə ayrılır, hər nöqtənin yanında isə "
    "illik məbləğ və əvvəlki ilə nisbətən artım/faiz dəyişikliyi qeyd olunur. Forecast yalnız il, group, departament, "
    "filial və vəzifə filtrlərinə uyğun olaraq hesablanır, year filterindən asılı deyil ki, trendi itirməyəsiniz._"
)
