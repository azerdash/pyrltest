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

# Group colors üçün əsas palet
GROUP_COLORS = {
    "Head Office": "#002855",  # deep navy
    "IT": "#00509e",           # royal blue
    "Branch": "#7fb3ff",       # light blue
}

# 🔸 Group comparison üçün daha zövqlü pastel palet
GROUP_COLORS_LIGHT = {
    "Head Office": "#f8dce6",  # light blue
    "IT": "#ede7ec",           # light red/pink
    "Branch": "#d3defb",       # light green
}

# Department table payment types
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

# Chart selectbox-ları üçün ödəniş növləri (AZ)
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

    Digər ödənişlər üçün:
      - Layihə/Təlim/Birdəfəlik/Qalan digər sub-komponentlərinin həm məbləği, həm də artım faizi
        total və per-employee chartları üçün,
      - eyni sub-komponentlər üzrə əməkdaş sayı və artım faizi employees chartı üçün
        hover məlumatında göstərilir.
    """
    if df_scope.empty:
        empty = pd.DataFrame(columns=[COL_YEAR, "Component", "value", "growth_pct"])
        return empty.copy(), empty.copy(), empty.copy()

    # ---- 1) İl üzrə komponent məbləğləri ----
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
        - year_components_amount["other_proj"]
        - year_components_amount["other_train"]
        - year_components_amount["other_once"]
    )

    # İllərə görə sırala və sub-komponentlər üçün growth hesabla
    year_components_amount = year_components_amount.sort_values(COL_YEAR)
    for sub in ["other_proj", "other_train", "other_once", "other_rest"]:
        year_components_amount[f"{sub}_growth"] = (
            year_components_amount[sub].pct_change() * 100
        )

    # ---- 1a) Komponent total-ları (AZN) ----
    comp_total_long = year_components_amount.melt(
        id_vars=[
            COL_YEAR,
            "other_proj", "other_train", "other_once", "other_rest",
            "other_proj_growth", "other_train_growth",
            "other_once_growth", "other_rest_growth",
        ],
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
        tmp = df_scope[df_scope[col_name] > 0]
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

    # Sub-komponentlər üzrə əməkdaş sayı (Digər ödənişlər üçün hover-də istifadə olunur)
    sub_map_emp = {
        "other_proj": COL_BONUS_PROJECT,
        "other_train": COL_BONUS_TRAINING,
        "other_once": COL_BONUS_OTHER,
        "other_rest": "Remaining other",
    }
    sub_emp_year = None
    for sub_key, sub_col in sub_map_emp.items():
        if sub_col not in df_scope.columns:
            continue
        tmp = df_scope[df_scope[sub_col] > 0]
        if tmp.empty:
            continue
        monthly_emp = (
            tmp.groupby([COL_YEAR, COL_MONTH])[COL_ID]
            .nunique()
            .reset_index(name=f"{sub_key}_emp")
        )
        avg_emp = (
            monthly_emp
            .groupby(COL_YEAR, as_index=False)[f"{sub_key}_emp"]
            .mean()
        )
        avg_emp = avg_emp.sort_values(COL_YEAR)
        avg_emp[f"{sub_key}_emp_growth"] = avg_emp[f"{sub_key}_emp"].pct_change() * 100
        if sub_emp_year is None:
            sub_emp_year = avg_emp
        else:
            sub_emp_year = sub_emp_year.merge(avg_emp, on=COL_YEAR, how="outer")

    if sub_emp_year is not None and not comp_emp_long.empty:
        comp_emp_long = comp_emp_long.merge(sub_emp_year, on=COL_YEAR, how="left")
        for sub_key in ["other_proj", "other_train", "other_once", "other_rest"]:
            if f"{sub_key}_emp" in comp_emp_long.columns:
                comp_emp_long[sub_key] = comp_emp_long[f"{sub_key}_emp"]
            if f"{sub_key}_emp_growth" in comp_emp_long.columns:
                comp_emp_long[f"{sub_key}_growth"] = comp_emp_long[f"{sub_key}_emp_growth"]

    # ---- 3) Per-employee breakdown ----
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
        [
            COL_YEAR,
            COL_SALARY, COL_BONUS, COL_EOY, COL_VACATION, COL_OTHER2,
            "other_proj", "other_train", "other_once", "other_rest",
        ]
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

    per_emp_breakdown = per_emp_breakdown.sort_values(COL_YEAR)
    for sub in ["other_proj", "other_train", "other_once", "other_rest"]:
        per_emp_breakdown[f"{sub}_growth"] = (
            per_emp_breakdown[sub].pct_change() * 100
        )

    per_emp_long = per_emp_breakdown.melt(
        id_vars=[
            COL_YEAR,
            "other_proj", "other_train", "other_once", "other_rest",
            "other_proj_growth", "other_train_growth",
            "other_once_growth", "other_rest_growth",
        ],
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
    if df_year.empty:
        return None

    df_plot = df_year.sort_values(COL_YEAR).copy()
    diff_col = f"{value_col}_diff"
    df_plot[diff_col] = df_plot[value_col].diff()

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

    for _, row in df_plot.iterrows():
        growth = row.get(growth_col)
        diff = row.get(diff_col)

        if pd.isna(growth) or not np.isfinite(growth):
            continue
        if pd.isna(diff) or not np.isfinite(diff):
            continue

        arrow = "▲" if diff > 0 else "▼" if diff < 0 else ""
        diff_text = fmt_amount(diff)
        label = fmt_growth(growth)
        label += f"<br>{arrow} {diff_text}"

        fig.add_annotation(
            x=row[COL_YEAR],
            y=row[value_col] / 2,
            text=label,
            showarrow=False,
            font=dict(color="white", size=11),
            borderwidth=0,
            align="center",
        )

    return fig


def create_breakdown_line(df_long: pd.DataFrame, title: str, unit: str = "AZN"):
    """
    Breakdown line chart with AZ component labels.
    Hover-da:
      - Bütün komponentlər üçün: dəyər + artım faizi (tam ədəd),
      - Digər ödənişlər üçün əlavə olaraq Layihə/Təlim/Birdəfəlik/Qalan digər
        sub-komponentlərinin həm dəyəri, həm də artım faizi göstərilir.
    """
    if df_long.empty:
        return None

    df_plot = df_long.copy()
    df_plot["year_str"] = df_plot[COL_YEAR].astype(str)

    # Line üzərindəki label – 1 decimal qalır
    def make_label(row):
        g = row.get("growth_pct")
        if pd.notna(g) and np.isfinite(g):
            return f"{fmt_amount(row['value'])} ({fmt_growth(g)})"
        else:
            return fmt_amount(row["value"])

    df_plot["label"] = df_plot.apply(make_label, axis=1)

    # 🔹 Hover üçün faizləri tam ədədə yuvarlaq
    df_plot["growth_int"] = df_plot["growth_pct"].round().astype("float")

    sub_cols_full = [
        "other_proj", "other_proj_growth",
        "other_train", "other_train_growth",
        "other_once", "other_once_growth",
        "other_rest", "other_rest_growth",
    ]
    has_sub = all(col in df_plot.columns for col in sub_cols_full)

    if has_sub:
        df_plot["other_proj_growth_int"] = df_plot["other_proj_growth"].round().astype("float")
        df_plot["other_train_growth_int"] = df_plot["other_train_growth"].round().astype("float")
        df_plot["other_once_growth_int"] = df_plot["other_once_growth"].round().astype("float")
        df_plot["other_rest_growth_int"] = df_plot["other_rest_growth"].round().astype("float")

    custom_cols = ["growth_int"]
    if has_sub:
        custom_cols += [
            "other_proj", "other_proj_growth_int",
            "other_train", "other_train_growth_int",
            "other_once", "other_once_growth_int",
            "other_rest", "other_rest_growth_int",
        ]

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

    # Hover templates – faizlər tam ədəd
    for trace in fig.data:
        name = trace.name
        if name == "Digər ödənişlər" and has_sub and len(custom_cols) >= 9:
            trace.hovertemplate = (
                "İl=%{x}<br>"
                "%{fullData.name}=%{y:,.0f} " + unit + "<br>"
                "• Layihə bonusu=%{customdata[1]:,.0f} " + unit + " (%{customdata[2]:+.0f}%)<br>"
                "• Təlim bonusu=%{customdata[3]:,.0f} " + unit + " (%{customdata[4]:+.0f}%)<br>"
                "• Digər birdəfəlik bonus=%{customdata[5]:,.0f} " + unit + " (%{customdata[6]:+.0f}%)<br>"
                "• Qalan digər=%{customdata[7]:,.0f} " + unit + " (%{customdata[8]:+.0f}%)<br>"
                "<extra></extra>"
            )
        else:
            trace.hovertemplate = (
                "İl=%{x}<br>"
                "%{fullData.name}=%{y:,.0f} " + unit + "<br>"
                "Artım=%{customdata[0]:+.0f}%"
                "<extra></extra>"
            )

    return fig


def render_breakdown_row(df_scope: pd.DataFrame,
                         row_title_prefix: str):
    comp_total_long, comp_emp_long, per_emp_long = compute_breakdowns(df_scope)

    with st.spinner("Loading charts..."):
        c1, c2, c3 = st.columns(3)

        with c1:
            fig = create_breakdown_line(
                comp_total_long,
                title=f"{row_title_prefix} – total payments by component",
                unit="AZN",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data for this chart with current filters.")

        with c2:
            fig = create_breakdown_line(
                comp_emp_long,
                title=f"{row_title_prefix} – employees receiving each component",
                unit="employees",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data for this chart with current filters.")

        with c3:
            fig = create_breakdown_line(
                per_emp_long,
                title=f"{row_title_prefix} – per-employee amount by component",
                unit="AZN",
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
    avg_monthly: bool = False,
) -> pd.DataFrame:
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

        # Orta aylıq gəlir üçün ay sayına bölək (yalnız amount və per_emp modları)
        if avg_monthly and mode in ("amount", "per_emp"):
            active = (
                df_metric[df_metric[base_col] > 0]
                .groupby([COL_REGION, COL_DEPT_GROUP, COL_NAME, COL_YEAR])[COL_MONTH]
                .nunique()
                .reset_index(name="active_months")
            )
            m_df = m_df.merge(
                active,
                on=[COL_REGION, COL_DEPT_GROUP, COL_NAME, COL_YEAR],
                how="left",
            )
            m_df["value"] = m_df["value"] / m_df["active_months"].replace(0, np.nan)
            m_df.drop(columns=["active_months"], inplace=True)

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
    if metrics_group.empty:
        return None

    df_plot = metrics_group.copy()

    def make_label(row):
        g = row.get(growth_col)
        if pd.notna(g) and np.isfinite(g):
            return f"{fmt_amount(row[value_col])}<br>({fmt_growth(g)})"
        else:
            return fmt_amount(row[value_col])

    df_plot["label"] = df_plot.apply(make_label, axis=1)

    fig = px.bar(
        df_plot,
        x=COL_YEAR,
        y=value_col,
        color=COL_REGION,
        barmode="relative",  # stacked
        text="label",
        color_discrete_map=GROUP_COLORS_LIGHT,
    )

    yearly_sum = df_plot.groupby(COL_YEAR)[value_col].sum()
    max_val = yearly_sum.max() if not yearly_sum.empty else 0
    min_val = min(0, yearly_sum.min()) if not yearly_sum.empty else 0

    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=11, color="black"),
        cliponaxis=False,
    )

    fig.update_layout(
        uniformtext_minsize=11,
        uniformtext_mode="show",
    )

    fig.update_xaxes(
        title_text="",
        showgrid=False,
        tickmode="array",
        tickvals=sorted(df_plot[COL_YEAR].unique()),
        ticktext=[str(y) for y in sorted(df_plot[COL_YEAR].unique())],
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
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
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

# Filtrlər
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

filtered = base_filtered.copy()
if selected_regions:
    filtered = filtered[filtered[COL_REGION].isin(selected_regions)]

if filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

# =========================================================
# 6. KEY METRICS BY YEAR
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
    "_Bu blok seçdiyiniz payment növü üzrə illər boyu üç əsas göstəricini göstərir: "
    "solda illik ümumi məbləğ, ortada həmin paymenti alan orta aylıq əməkdaş sayı, sağda isə bir əməkdaşa "
    "düşən orta məbləğ. Sütunun üstündə həmin ilin məbləği, sütunun içində isə əvvəlki ilə nisbətən artım/faiz "
    "və məbləğ fərqi görünür. Filtrlərlə (il, ay, group, departament, filial kurasiyası, vəzifə) işləyərək "
    "konkret seqment üzrə trendi ayrıca təhlil etmək mümkündür._"
)

# =========================================================
# 7. BREAKDOWN BY COMPONENTS (ALL GROUPS)
# =========================================================
st.subheader("Breakdown by components (all groups)")

render_breakdown_row(filtered, "Ümumi")

st.markdown(
    "_Bu sətirdə bütün group-lar (Head Office, IT, Branch birlikdə) üzrə Total payment müxtəlif komponentlərə "
    "bölünərək göstərilir. Soldakı qrafikdə hər komponent üzrə illik ümumi məbləğ, ortadakı qrafikdə həmin komponentdən "
    "aylıq faydalanan əməkdaşların orta sayı, sağdakı qrafikdə isə bir əməkdaşa düşən orta məbləğ görünür. "
    "Məzuniyyət ayrıca xətt kimi çıxarılıb, Digər ödənişlər xəttinə hover edəndə isə Layihə bonusu, Təlim bonusu, "
    "Birdəfəlik digər bonus və Qalan digər hissələri həm məbləğ, həm də əvvəlki ilə nisbətən artım faizi ilə birlikdə "
    "ayrıca göstərilir. Hər nöqtənin label və hover hissəsində komponent üzrə artım/faiz dəyişikliyi də görünür. "
    "EOY yalnız faktiki ödəniş olduğu illərdə xəttə əlavə edilir; seçilmiş periodda 0 olduqda qrafikdə yer tutmaması "
    "üçün gizlədilir._"
)

# =========================================================
# 8. GROUP COMPARISON – STACKED COLUMN
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
    "_Bu blokda Head Office, IT və Branch qrupları üzərində seçilmiş payment növü üzrə illik göstəricilər "
    "stacked column qrafiklərlə müqayisə olunur. Solda ümumi məbləğ, ortada həmin paymenti alan orta "
    "aylıq əməkdaş sayı, sağda isə bir əməkdaşa düşən orta məbləğ göstərilir. Hər ilin sütununda 3 group pastel "
    "rəng çalarlarında yığılmış şəkildə görünür, stack-lərin içində isə həmin group üzrə məbləğ və əvvəlki ilə "
    "nisbətən artım/faiz dəyəri iki sətirdə göstərilir. Bu blokda Group filteri deaktivdir (bütün group-lar "
    "mütləq göstərilir), lakin il, ay, departament, curator və vəzifə filterləri tətbiq olunur._"
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
                    unit="AZN",
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
                    unit="employees",
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
                    unit="AZN",
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "_Bu bölmə Breakdown by components məntiqini hər group üçün ayrıca göstərir: Head Office, IT və Branch "
    "sətir üzrə eyni struktura malik qrafiklərlə yanaşı yerləşdirilib. İlk sətirdə hər group üçün komponentlər üzrə "
    "ümumi məbləğ, ikinci sətirdə həmin komponentdən faydalanan əməkdaş sayı, üçüncü sətirdə isə bir əməkdaşa düşən "
    "orta məbləğ göstərilir. Hover zamanı Digər ödənişlər üzrə Layihə/Təlim/Birdəfəlik/Qalan digər hissələrinin "
    "də həm dəyər, həm də artım faizi ayrıca göstərilir._"
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
    "_Bu cədvəldə Group (Head Office / IT / Branch) və Department səviyyəsində seçdiyiniz payment növləri üzrə "
    "illər boyu üç ölçüdən birini görə bilərsiniz: Total amount paid, Average employees paid və Amount per employee. "
    "Hər il üçün artım/faiz ayrıca sütunda verilir; mənfi göstəricilər tünd yaşıl, müsbət göstəricilər tünd qırmızı "
    "rənglə vurğulanır. Cədvəli sort edib hansı departamentlərin daha sürətli artdığını və ya azaldığını rahat görə "
    "bilərsiniz. Bu filterlər yalnız department-level cədvələ aiddir._"
)

# =========================================================
# 11. EMPLOYEE-LEVEL TABLE
# =========================================================
st.subheader("Employee-level table")

emp_dept_options = sorted(df_table_base[COL_DEPT_GROUP].dropna().unique())
emp_selected_depts = st.multiselect(
    "Departments (for employee table)",
    options=emp_dept_options,
    default=emp_dept_options,
)

emp_avg_monthly = st.checkbox(
    "Show average monthly income (employee table)",
    value=True,
)

df_emp_scope = df_table_base.copy()
if emp_selected_depts:
    df_emp_scope = df_emp_scope[df_emp_scope[COL_DEPT_GROUP].isin(emp_selected_depts)]

emp_table = build_employee_metric_table(
    df_emp_scope,
    table_metric_keys,
    table_selected_years,
    table_mode,
    avg_monthly=emp_avg_monthly,
)

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
    "_Bu cədvəl eyni metrikləri əməkdaş səviyyəsində göstərir. Hər sətrdə Group, Department və Employee (S.A.A) "
    "görünür. Yuxarıdakı payment növləri və ölçü (total amount, average employees, amount per employee) seçiminə "
    "uyğun olaraq illər üzrə dəyərlər və artım faizləri hesablanır. "
    "\"Show average monthly income\" aktivdirsə, amount və per-employee rejimlərində hər il üçün cəmi həmin "
    "əməkdaşın həmin il aktiv olduğu ayların sayına bölünür və beləliklə fərqli müddətlərdə çalışan əməkdaşlar "
    "orta aylıq gəlir əsasında müqayisə edilir._"
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

monthly_series[COL_MONTH] = pd.Categorical(
    monthly_series[COL_MONTH],
    categories=MONTH_ORDER,
    ordered=True,
)

monthly_series = monthly_series.sort_values([COL_MONTH, COL_YEAR])

monthly_series["growth_pct"] = (
    monthly_series
    .groupby(COL_MONTH)["metric_value"]
    .pct_change() * 100
)
# Hover üçün tam ədəd
monthly_series["growth_int"] = monthly_series["growth_pct"].round().astype("float")

def make_monthly_label(row):
    g = row.get("growth_pct")
    if pd.notna(g) and np.isfinite(g):
        return f"{fmt_amount(row['metric_value'])}<br>({fmt_growth(g)})"
    else:
        return fmt_amount(row["metric_value"])

monthly_series["label"] = monthly_series.apply(make_monthly_label, axis=1)

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
            text="label",
            category_orders={COL_MONTH: MONTH_ORDER},
            custom_data=["growth_int"],
        )
        fig_m.update_traces(
            line=dict(width=3),
            textposition="top center",
            textfont=dict(size=11, color="black"),
            cliponaxis=False,
            hovertemplate=(
                "Year=%{fullData.name}<br>"
                "Month=%{x}<br>"
                "Amount=%{y:,.0f} AZN<br>"
                "Growth=%{customdata[0]:+.0f}%"
                "<extra></extra>"
            ),
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
    "_Bu qrafik seçilmiş payment növü üzrə illər və filtrlər çərçivəsində aylıq dinamikaları göstərir. Hər nöqtənin "
    "üstündə həmin ay üzrə məbləğ, mötərizədə isə eyni ayın əvvəlki ilə nisbətən artım/faiz dəyişikliyi verilir. "
    "Hover hissəsində growth dəyərləri tam faizlərlə (məsələn, +12%) göstərilir. X oxunda Yanvar–Dekabr ardıcıllığında "
    "aylar, rənglərlə isə illər fərqləndirilir._"
)

# =========================================================
# 13. CARI SUPER GROSS BY YEAR
# =========================================================
st.subheader("Cari Super Gross by year")

csg_df = base_filtered.copy()

if COL_CURR_SUPER_GROSS not in csg_df.columns:
    st.info("Cari Super Gross column not found in data.")
else:
    csg_year = (
        csg_df
        .groupby(COL_YEAR, as_index=False)[COL_CURR_SUPER_GROSS]
        .sum()
        .rename(columns={COL_CURR_SUPER_GROSS: "total_csg"})
    )

    if csg_year.empty:
        st.info("No data for Cari Super Gross with current filters.")
    else:
        csg_year = add_growth_by_year(csg_year, "total_csg")
        fig_csg = create_bar_with_growth(
            csg_year,
            value_col="total_csg",
            growth_col="total_csg_growth_pct",
            title="Total Cari Super Gross by year (bank-wide)",
        )
        if fig_csg:
            st.plotly_chart(fig_csg, use_container_width=True)
        else:
            st.info("No data for Cari Super Gross with current filters.")

st.markdown(
    "_Bu qrafik bank üzrə Cari Super Gross məbləğini illər üzrə göstərir. Cari Super Gross – əməkdaşlara ödənilən "
    "Total payment-lə yanaşı, bank tərəfindən həmin məbləğlər üzrə ödənilən vergiləri də əhatə edən ümumi xərcdir. "
    "Sütunun üstündə hər il üçün ümumi Cari Super Gross, içində isə əvvəlki ilə nisbətən artım/faiz və məbləğ fərqi "
    "görünür. Filterlər tətbiq olunduqda qrafik yalnız seçilmiş kəsim üzrə ümumi Cari Super Gross-u göstərir._"
)
