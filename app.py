import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Zomato Food Recommendation Engine",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

ZOMATO_RED = "#e23744"

# ----------------------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #fafafa; }}
    .zomato-badge {{
        background-color: {ZOMATO_RED};
        color: white;
        padding: 6px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 20px;
        display: inline-block;
        letter-spacing: 0.5px;
    }}
    .hero-title {{
        font-size: 42px;
        font-weight: 800;
        color: #1a1a1a;
        margin-bottom: 0px;
    }}
    .hero-sub {{
        color: #6b6b6b;
        font-size: 16px;
        margin-top: 4px;
    }}
    .food-card {{
        background: white;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        border: 1px solid #eee;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.15s ease;
    }}
    .food-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(226,55,68,0.15);
    }}
    .food-emoji {{ font-size: 40px; }}
    .food-name {{ font-weight: 700; font-size: 15px; margin: 6px 0 2px; }}
    .food-price {{ color: {ZOMATO_RED}; font-weight: 700; font-size: 14px; }}
    .confidence-pill {{
        background: #fff3ea;
        color: #b45309;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
        margin-top: 6px;
    }}
    .cart-card {{
        background: white;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #eee;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .cart-row {{
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px dashed #eee;
        font-size: 15px;
    }}
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #eee;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        text-align: center;
    }}
    .metric-value {{ font-size: 26px; font-weight: 800; color: {ZOMATO_RED}; }}
    .metric-label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# DEFAULT DATA
# ----------------------------------------------------------------------------
DEFAULT_TRANSACTIONS = [
    ["Veg Biryani", "Raita", "Coke"],
    ["Veg Biryani", "Raita"],
    ["Veg Biryani", "Masala Papad", "Coke"],
    ["Veg Biryani", "Raita", "Masala Papad"],
    ["Chicken Biryani", "Coke"],
    ["Chicken Biryani", "Raita"],
    ["Pizza", "Garlic Bread", "Pepsi"],
    ["Pizza", "Garlic Bread"],
    ["Burger", "Fries", "Coke"],
    ["Burger", "Fries"],
    ["Burger", "Coke"],
    ["Veg Biryani", "Coke"],
    ["Veg Biryani", "Raita", "Coke", "Masala Papad"],
    ["Pizza", "Pepsi"],
    ["Chicken Biryani", "Coke", "Raita"],
    ["Burger", "Fries", "Pepsi"],
    ["Veg Biryani", "Raita", "Coke"],
    ["Veg Biryani", "Raita"],
    ["Pizza", "Garlic Bread", "Pepsi"],
    ["Burger", "Fries"],
]

ITEM_EMOJI = {
    "Veg Biryani": "🍛", "Chicken Biryani": "🍗", "Raita": "🥣",
    "Coke": "🥤", "Pepsi": "🥤", "Masala Papad": "🫓",
    "Pizza": "🍕", "Garlic Bread": "🥖", "Burger": "🍔", "Fries": "🍟",
}
ITEM_PRICE = {
    "Veg Biryani": 249, "Chicken Biryani": 299, "Raita": 49,
    "Coke": 40, "Pepsi": 40, "Masala Papad": 69,
    "Pizza": 349, "Garlic Bread": 129, "Burger": 179, "Fries": 99,
}


def emoji_for(item):
    return ITEM_EMOJI.get(item, "🍽️")


def price_for(item):
    return ITEM_PRICE.get(item, 99)


# ----------------------------------------------------------------------------
# CACHE HELPERS
# ----------------------------------------------------------------------------
@st.cache_data
def encode_transactions(transactions):
    te = TransactionEncoder()
    encoded = te.fit(transactions).transform(transactions)
    return pd.DataFrame(encoded, columns=te.columns_)


@st.cache_data
def get_frequent_itemsets(df, min_support):
    itemsets = apriori(df, min_support=min_support, use_colnames=True)
    itemsets["length"] = itemsets["itemsets"].apply(len)
    itemsets.sort_values(by="support", ascending=False, inplace=True)
    return itemsets


@st.cache_data
def get_rules(frequent_itemsets, min_confidence):
    if frequent_itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules.sort_values(by="confidence", ascending=False, inplace=True)
    return rules


def sets_to_str(col):
    return col.apply(lambda x: ", ".join(sorted(list(x))))


def recommend_for_cart(rules, cart_items, top_n=5):
    """Aggregate recommendations for every item currently in the cart."""
    if rules.empty or not cart_items:
        return pd.DataFrame()
    unique_cart = set(cart_items)
    matches = rules[rules["antecedents"].apply(lambda x: bool(set(x) & unique_cart))].copy()
    if matches.empty:
        return matches
    matches["rec_item"] = matches["consequents"].apply(lambda x: next(iter(x)) if len(x) == 1 else ", ".join(x))
    matches = matches[matches["consequents"].apply(lambda x: not (set(x) <= unique_cart))]
    matches.sort_values(by="confidence", ascending=False, inplace=True)
    matches.drop_duplicates(subset=["rec_item"], keep="first", inplace=True)
    return matches.head(top_n)


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "cart" not in st.session_state:
    st.session_state.cart = []

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Apriori settings")

data_choice = st.sidebar.radio("Data source", ["Built-in sample orders", "Upload my own CSV"], index=0)
transactions = DEFAULT_TRANSACTIONS

if data_choice == "Upload my own CSV":
    st.sidebar.caption("One order per row, items in columns.")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded, header=None)
        transactions = [[str(v) for v in row if pd.notna(v)] for row in raw.values.tolist()]
        st.sidebar.success(f"Loaded {len(transactions)} orders.")
    else:
        st.sidebar.info("No file uploaded — using sample data.")

min_support = st.sidebar.slider("Minimum support", 0.05, 0.9, 0.20, 0.05)
min_confidence = st.sidebar.slider("Minimum confidence", 0.05, 0.9, 0.50, 0.05)
top_n = st.sidebar.slider("Recommendations to show", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Support** — how often an itemset appears.

**Confidence** — P(Y | X): how often Y is bought when X is bought.

**Lift** — how much more likely X & Y occur together vs. independently.
"""
)

# ----------------------------------------------------------------------------
# PROCESS DATA
# ----------------------------------------------------------------------------
df = encode_transactions(transactions)
frequent_itemsets = get_frequent_itemsets(df, min_support)
rules = get_rules(frequent_itemsets, min_confidence)
all_items = sorted(df.columns.tolist())

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
head_left, head_right = st.columns([5, 1])
with head_left:
    st.markdown('<div class="hero-title">🍽️ Zomato food recommendation engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Powered by the <b>Apriori algorithm</b> — association rule mining for smart cross-sell recommendations</div>',
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown('<div class="zomato-badge">zomato</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
for col, label, value in zip(
    [m1, m2, m3, m4],
    ["Total orders", "Unique items", "Itemsets found", "Rules found"],
    [len(transactions), df.shape[1], len(frequent_itemsets), len(rules)],
):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🛒 Try it — cart demo", "📦 Frequent itemsets", "🔗 Association rules", "📊 Visualization", "🕸️ Item network"]
)

# ---- TAB 1: Cart demo ----
with tab1:
    st.markdown("#### Add an item to your cart")
    left, right = st.columns([3, 2])

    with left:
        if not all_items:
            st.warning("No items available.")
        else:
            selected_item = st.selectbox("What did the customer order?", all_items, label_visibility="collapsed")
            add_col, clear_col = st.columns(2)
            if add_col.button("➕ Add to cart", type="primary", use_container_width=True):
                st.session_state.cart.append(selected_item)
                st.rerun()
            if clear_col.button("🗑️ Clear cart", use_container_width=True):
                st.session_state.cart = []
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🧺 Your cart")
        if not st.session_state.cart:
            st.info("Cart is empty — add an item above to see live recommendations.")
        else:
            rows_html = ""
            total = 0
            for item in st.session_state.cart:
                p = price_for(item)
                total += p
                rows_html += (
                    f'<div class="cart-row"><span>{emoji_for(item)} &nbsp; {item}</span>'
                    f'<span>₹{p}</span></div>'
                )
            st.markdown(
                f'<div class="cart-card">{rows_html}'
                f'<div class="cart-row" style="border-bottom:none; font-weight:800; padding-top:10px;">'
                f'<span>Total</span><span>₹{total}</span></div></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### 🔔 You may also like")
        preview_items = list(st.session_state.cart)
        if all_items and selected_item and selected_item not in preview_items:
            preview_items = preview_items + [selected_item]
        rec = recommend_for_cart(rules, preview_items, top_n=top_n)
        if not preview_items:
            st.caption("Select an item to see recommendations.")
        elif rec.empty:
            st.warning("No recommendation at current thresholds — try lowering support/confidence.")
        else:
            cards_per_row = 2
            rows_of_recs = [rec.iloc[i:i + cards_per_row] for i in range(0, len(rec), cards_per_row)]
            for chunk in rows_of_recs:
                cols = st.columns(cards_per_row)
                for c, (_, row) in zip(cols, chunk.iterrows()):
                    item_name = row["rec_item"]
                    with c:
                        st.markdown(
                            f'<div class="food-card">'
                            f'<div class="food-emoji">{emoji_for(item_name)}</div>'
                            f'<div class="food-name">{item_name}</div>'
                            f'<div class="food-price">₹{price_for(item_name)}</div>'
                            f'<div class="confidence-pill">{row["confidence"]*100:.0f}% confidence · lift {row["lift"]:.2f}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button("Add ➕", key=f"add_{item_name}_{row.name}", use_container_width=True):
                            st.session_state.cart.append(item_name)
                            st.rerun()

# ---- TAB 2: Frequent Itemsets ----
with tab2:
    st.markdown("#### Frequent itemsets")
    if frequent_itemsets.empty:
        st.warning("No frequent itemsets found. Try lowering minimum support.")
    else:
        display_df = frequent_itemsets.copy()
        display_df["itemsets"] = sets_to_str(display_df["itemsets"])
        st.dataframe(
            display_df[["itemsets", "support", "length"]].reset_index(drop=True),
            use_container_width=True,
        )

# ---- TAB 3: Association Rules ----
with tab3:
    st.markdown("#### Association rules")
    if rules.empty:
        st.warning("No rules found. Try lowering minimum support/confidence.")
    else:
        display_rules = rules.copy()
        display_rules["antecedents"] = sets_to_str(display_rules["antecedents"])
        display_rules["consequents"] = sets_to_str(display_rules["consequents"])
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        st.dataframe(
            display_rules[cols]
            .style.background_gradient(cmap="Reds", subset=["confidence"])
            .format({"support": "{:.2f}", "confidence": "{:.2f}", "lift": "{:.2f}"}),
            use_container_width=True,
        )

# ---- TAB 4: Visualization ----
with tab4:
    st.markdown("#### Top frequent itemsets by support")
    if frequent_itemsets.empty:
        st.warning("No data to plot.")
    else:
        top = frequent_itemsets.copy()
        top["Items"] = sets_to_str(top["itemsets"])
        top = top.head(10)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(top["Items"], top["support"], color=ZOMATO_RED)
        ax.set_ylabel("Support")
        ax.set_title("Top frequent itemsets")
        plt.xticks(rotation=60, ha="right")
        st.pyplot(fig)

    if not rules.empty:
        st.markdown("#### Confidence vs lift")
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.scatter(rules["confidence"], rules["lift"], c=ZOMATO_RED, alpha=0.75, s=70)
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Lift")
        ax2.set_title("Rule confidence vs lift")
        st.pyplot(fig2)

# ---- TAB 5: Item Network ----
with tab5:
    st.markdown("#### Item association network")
    st.caption("Nodes = food items, arrows = association rules, arrow thickness = confidence strength.")
    if rules.empty:
        st.warning("No rules to draw. Try lowering minimum support/confidence.")
    else:
        G = nx.DiGraph()
        for _, row in rules.iterrows():
            for a in row["antecedents"]:
                for c in row["consequents"]:
                    G.add_edge(a, c, weight=row["confidence"])

        pos = nx.spring_layout(G, seed=42, k=1.2)
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        weights = [G[u][v]["weight"] * 4 for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_color="#ffe3e6", node_size=2200, edgecolors=ZOMATO_RED, linewidths=2, ax=ax3)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax3)
        nx.draw_networkx_edges(
            G, pos, width=weights, edge_color=ZOMATO_RED, alpha=0.6,
            arrowsize=18, connectionstyle="arc3,rad=0.08", ax=ax3,
        )
        ax3.axis("off")
        st.pyplot(fig3)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Made with ❤️ using Streamlit, pandas, mlxtend & networkx | Zomato Food Recommendation Project")
