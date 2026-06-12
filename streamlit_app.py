"""
Interactive Streamlit app for the 50 Startups CRISP-DM regression project.

Run locally:
    streamlit run streamlit_app.py

Tabs:
    1. Tutorial slides  - browse the hand-drawn deck with notes
    2. Data exploration - dataset, statistics, correlations
    3. Model comparison - the five feature experiments (Models A-E)
    4. Profit predictor - interactive what-if prediction
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = os.path.join("sources", "50_Startups.csv")
SLIDES_DIR = os.path.join("startup-presentation-video-pptx", "assets", "slides")
NUMERICAL_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
TARGET = "Profit"

SLIDE_NOTES = {
    1: ("預測 50 家新創的獲利", "Kaggle 50 Startups 資料集：用支出結構與州別預測 Profit。CRISP-DM × Scikit-Learn。"),
    2: ("商業目標：解碼獲利引擎", "把公司想成機器：研發、行銷、行政三種投入，哪一根槓桿真正推動獲利？"),
    3: ("結構藍圖：CRISP-DM", "商業理解 → 資料理解 → 資料準備 → 建模 → 評估 → 結論。人類與機器兩條路平行進行。"),
    4: ("Scikit-Learn 工具箱", "Pipeline + ColumnTransformer；State 用 OneHotEncoder(drop='first') 避免虛擬變數陷阱；80/20 切分。"),
    5: ("通往真相的兩條路", "人類路徑（商業優先序）vs 機器路徑（特徵選擇演算法）。兩條路若收斂，結論就可信。"),
    6: ("路徑一：人類引導實驗", "Model A（只用 R&D）→ B（+行銷）→ C（+行政）→ D（全特徵）→ E（只用州）。"),
    7: ("路徑二：演算法特徵選擇", "Wrapper（SFS、RFE）、Embedded（Lasso、隨機森林）、Filter（SelectKBest）五種方法獨立評選。"),
    8: ("收斂", "所有路徑的單特徵首選都是 R&D Spend：RMSE 7,714、R² 0.9265、Adjusted R² 0.9173。"),
    9: ("複雜度的代價", "特徵越多，測試誤差反而越大——50 筆小樣本下多餘特徵就是雜訊。"),
    10: ("特徵天平", "R&D 是核心（係數 ≈ 0.85）；行銷是輔助；行政不顯著（p=0.61）；State 樣本太小勿過度解讀。"),
    11: ("總結", "資料很小，商業洞察很大。選擇能說對故事的特徵，而不是堆疊特徵。"),
}

EXPERIMENTS = [
    ("Model A: R&D Spend Only", ["R&D Spend"], []),
    ("Model B: R&D + Marketing", ["R&D Spend", "Marketing Spend"], []),
    ("Model C: R&D + Marketing + Administration", ["R&D Spend", "Marketing Spend", "Administration"], []),
    ("Model D: All Features + State", ["R&D Spend", "Marketing Spend", "Administration", "State"], ["State"]),
    ("Model E: State Only", ["State"], ["State"]),
]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def make_pipeline(features, categorical):
    numeric = [f for f in features if f not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", "passthrough", numeric))
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical)
        )
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers)),
            ("regressor", LinearRegression()),
        ]
    )


def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


@st.cache_resource
def train_models():
    df = load_data()
    X = df[NUMERICAL_FEATURES + ["State"]]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models, rows = {}, []
    for name, features, categorical in EXPERIMENTS:
        pipe = make_pipeline(features, categorical)
        pipe.fit(X_tr[features], y_tr)
        pred = pipe.predict(X_te[features])
        r2 = r2_score(y_te, pred)
        n_pred = pipe.named_steps["preprocessor"].transform(X_te[features]).shape[1]
        rows.append(
            {
                "Model": name,
                "Features": ", ".join(features),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "MAE": float(mean_absolute_error(y_te, pred)),
                "R²": r2,
                "Adjusted R²": adjusted_r2(r2, len(y_te), n_pred),
            }
        )
        models[name] = (pipe, features)
    results = (
        pd.DataFrame(rows).sort_values("Adjusted R²", ascending=False).reset_index(drop=True)
    )
    return models, results, (X_te, y_te)


st.set_page_config(page_title="50 Startups 獲利解碼", page_icon="🚀", layout="wide")
st.title("🚀 Decoding Startup Profit — 50 Startups 互動分析")
st.caption("CRISP-DM × Scikit-Learn · Multiple Linear Regression · HW6")

tab_tutorial, tab_data, tab_models, tab_predict = st.tabs(
    ["📖 教學投影片", "📊 資料探索", "⚖️ 模型比較", "🎯 互動預測"]
)

# ---------------------------------------------------------------- tutorial
with tab_tutorial:
    col_img, col_note = st.columns([2.2, 1])
    with col_note:
        page = st.slider("投影片", 1, 11, 1)
        title, note = SLIDE_NOTES[page]
        st.subheader(f"{page:02d} · {title}")
        st.write(note)
        st.progress(page / 11)
    with col_img:
        slide_path = os.path.join(SLIDES_DIR, f"slide-{page:02d}.png")
        if os.path.exists(slide_path):
            st.image(slide_path, width="stretch")
        else:
            st.warning(f"找不到投影片圖片：{slide_path}")

# ---------------------------------------------------------------- data
with tab_data:
    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("樣本數", len(df))
    c2.metric("平均獲利", f"${df[TARGET].mean():,.0f}")
    c3.metric("最高獲利", f"${df[TARGET].max():,.0f}")
    c4.metric("R&D ↔ Profit 相關", f"{df['R&D Spend'].corr(df[TARGET]):.3f}")

    st.dataframe(df, width="stretch", height=260)

    left, right = st.columns(2)
    with left:
        feature = st.selectbox("選擇特徵看散佈圖", NUMERICAL_FEATURES)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for state, group in df.groupby("State"):
            ax.scatter(group[feature], group[TARGET], label=state, alpha=0.75)
        ax.set_xlabel(feature)
        ax.set_ylabel("Profit")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.35)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.write("**相關係數矩陣**")
        corr = df[NUMERICAL_FEATURES + [TARGET]].corr()
        st.dataframe(
            corr.style.background_gradient(cmap="RdYlBu", vmin=-1, vmax=1).format("{:.3f}"),
            width="stretch",
        )
        st.write("**各州平均獲利**")
        st.bar_chart(df.groupby("State")[TARGET].mean())

# ---------------------------------------------------------------- models
with tab_models:
    models, results, (X_te, y_te) = train_models()
    st.write("五組特徵實驗，同一個 80/20 切分（random_state=42），按 Adjusted R² 排序：")
    st.dataframe(
        results.style.format(
            {"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R²": "{:.4f}", "Adjusted R²": "{:.4f}"}
        ).highlight_max(subset=["Adjusted R²"], color="#c8e6c9"),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_df = results.sort_values("Adjusted R²")
        ax.barh(plot_df["Model"], plot_df["Adjusted R²"], color="#2f6f8f")
        ax.set_xlabel("Adjusted R²")
        ax.set_title("Adjusted R²（越高越好）")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        pick = st.selectbox("實際 vs 預測", list(models.keys()))
        pipe, feats = models[pick]
        pred = pipe.predict(X_te[feats])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(y_te, pred, color="#6a8f3f", alpha=0.85)
        lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
        ax.plot(lims, lims, "k--", lw=1.5)
        ax.set_xlabel("Actual Profit")
        ax.set_ylabel("Predicted Profit")
        ax.grid(True, linestyle="--", alpha=0.35)
        st.pyplot(fig)
        plt.close(fig)

    integrated = os.path.join("outputs", "figures", "feature_selection_integrated_comparison.png")
    if os.path.exists(integrated):
        with st.expander("🔍 四種特徵選擇分析的整合比較"):
            st.image(integrated, width="stretch")

# ---------------------------------------------------------------- predict
with tab_predict:
    models, results, _ = train_models()
    st.write("拖動滑桿，看看資源配置如何改變預測獲利：")
    df = load_data()

    c1, c2 = st.columns(2)
    with c1:
        rd = st.slider("R&D Spend（研發支出）", 0, 170_000, 75_000, step=1_000)
        marketing = st.slider("Marketing Spend（行銷支出）", 0, 475_000, 210_000, step=5_000)
    with c2:
        admin = st.slider("Administration（行政支出）", 50_000, 185_000, 120_000, step=1_000)
        state = st.selectbox("State（州別）", sorted(df["State"].unique()))

    row = pd.DataFrame(
        [{"R&D Spend": rd, "Administration": admin, "Marketing Spend": marketing, "State": state}]
    )

    best_pipe, best_feats = models["Model A: R&D Spend Only"]
    full_pipe, full_feats = models["Model D: All Features + State"]
    p_best = float(best_pipe.predict(row[best_feats])[0])
    p_full = float(full_pipe.predict(row[full_feats])[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("🏆 最佳模型預測（Model A：只看 R&D）", f"${p_best:,.0f}")
    m2.metric("全特徵模型預測（Model D）", f"${p_full:,.0f}", delta=f"{p_full - p_best:,.0f} vs Model A")
    rmse_a = float(results.loc[results["Model"] == "Model A: R&D Spend Only", "RMSE"].iloc[0])
    m3.metric("Model A 測試 RMSE（誤差幅度參考）", f"±${rmse_a:,.0f}")

    st.info(
        "💡 注意兩個模型的預測差異主要來自行政/行銷/州別——但測試集證明這些特徵"
        "提升不了準確度（Model A 的 Adjusted R² 0.917 仍是最高）。"
        "R&D 係數約 0.85：每多投入 1 元研發，預測獲利增加約 0.85 元。"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df["R&D Spend"], df[TARGET], alpha=0.55, color="#2f6f8f", label="50 家公司")
    xs = np.linspace(0, 170_000, 50)
    ys = best_pipe.predict(pd.DataFrame({"R&D Spend": xs}))
    ax.plot(xs, ys, color="#c24031", lw=2, label="Model A 回歸線")
    ax.scatter([rd], [p_best], s=160, color="#dfb85a", edgecolor="#2b2722", zorder=3, label="你的預測點")
    ax.set_xlabel("R&D Spend")
    ax.set_ylabel("Profit")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    st.pyplot(fig)
    plt.close(fig)
