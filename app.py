import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, QED, Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION & DESIGN CSS (LOOK MASTER DS)
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="AI Molecular Discovery Lab", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 20% 20%, #0d1b2a 0%, #010409 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Header Container */
    .header-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 40px; border-radius: 24px; margin-bottom: 30px;
        text-align: center; border-bottom: 4px solid #00d4ff;
    }

    /* Cartes de métriques */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px; padding: 24px; text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-8px); border-color: #00d4ff; }
    .metric-value { font-size: 2.2em; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.9em; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; }

    /* Titres Labo */
    .section-title {
        font-size: 1.5em; font-weight: 600; color: #00d4ff;
        border-left: 6px solid #00d4ff; padding-left: 20px; margin: 35px 0 20px 0;
    }

    /* Tabs Decoration */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 12px 12px 0 0; padding: 12px 24px; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] { background-color: #00d4ff !important; color: #000 !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 2. CHARGEMENT DES DONNÉES & FONCTIONS UTILES
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        return pd.read_csv('generated_molecules.csv'), pd.read_csv('my_filtered_molecules.csv'), pd.read_csv('molecules_ml_toxicity.csv')
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_resource
def load_model():
    try: return joblib.load('best_toxicity_model.pkl')
    except: return None

df_gen, df_filtered, df_tox = load_data()
best_model = load_model()

FEATURE_NAMES = [f'FP_bit_{i}' for i in range(1024)] + ['MolWt','MolLogP','TPSA','NumHBD','NumHBA','RotatableBonds','AromaticRings','HeavyAtoms','FractionCSP3','MolMR']

CF_MODIFICATIONS = [
    ('[N+:1](=[O:2])[O-:3]>>[OH:1]',   'Nitro → Hydroxyle'),
    ('[N+:1](=[O:2])[O-:3]>>[NH2:1]',  'Nitro → Amine'),
    ('[Cl:1]>>[F:1]',                   'Chlore → Fluor'),
    ('[Cl:1]>>[OH:1]',                  'Chlore → Hydroxyle'),
    ('[S:1][H]>>[O:1][H]',              'Thiol → Hydroxyle'),
    ('[N:1]=[N:2]>>[C:1]=[C:2]',       'Azo → Alcène'),
]

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    fp = GetMorganGenerator(radius=2, fpSize=1024).GetFingerprintAsNumPy(mol)
    props = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol), Descriptors.MolMR(mol)])
    return np.concatenate([fp, props])

# ══════════════════════════════════════════════════════════════
# 3. SIDEBAR & HEADER
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ FILTRES")
    qed_min = st.slider("QED minimum", 0.0, 1.0, 0.7, 0.01)
    sa_max = st.slider("SA Score max", 1.0, 5.0, 3.0, 0.1)
    tox_only = st.radio("Toxicité", ["Toutes", "Non-toxiques ✅", "Toxiques ❌"])
    n_show = st.slider("Molécules à afficher", 4, 32, 12, 4)
    st.markdown("---")
    st.info(f"📊 **Dataset :** {len(df_gen)} mol. générées")

st.markdown(f"""<div class="header-container">
    <h1 style='margin:0; font-size:2.8em; color:#00d4ff;'>🧬 DE NOVO MOLECULAR GENERATION</h1>
    <p style='margin:12px 0 0; color:#94a3b8; font-size:1.2em;'>Génération · Filtrage · Prédiction de Toxicité ML</p>
    <p style='color:#4a5568; font-size:0.9em; margin-top:8px;'>Master Big Data & DS | FSBM Hassan II Casablanca · 2025-2026</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 4. LES 5 ONGLETS
# ══════════════════════════════════════════════════════════════
if df_tox.empty:
    st.error("Données introuvables. Veuillez vérifier vos fichiers CSV.")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Vue globale", "🔬 Molécules", "🗺️ Espace latent", "☠️ Toxicité ML", "📋 Données"])

    # --- ONGLET 1 : VUE GLOBALE ---
    with tab1:
        st.markdown('<div class="section-title">Indicateurs de Performance</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for col, (v,l) in zip(cols, [("100%","Validité"),("69%","Unicité"),("100%","Nouveauté"),("0.823","QED moyen"),("2.667","SA Score")]):
            col.markdown(f'<div class="metric-card"><div class="metric-value">{v}</div><div class="metric-label">{l}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Comparaison vs État de l\'Art</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            df_lit = pd.DataFrame({'Modèle':['VAE','GAN','GDSS','🏆 MolMIM (NIM)'],'QED':[0.50, 0.55, 0.61, 0.82]})
            st.dataframe(df_lit, use_container_width=True, hide_index=True)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4)); fig.patch.set_alpha(0); ax.set_facecolor('none')
            ax.bar(df_lit['Modèle'], df_lit['QED'], color=['#1e293b','#1e293b','#1e293b','#00d4ff'])
            ax.tick_params(colors='white'); st.pyplot(fig)

    # --- ONGLET 2 : MOLÉCULES (GALLERY) ---
    with tab2:
        st.markdown('<div class="section-title">Candidats Moléculaires</div>', unsafe_allow_html=True)
        dv = df_tox[(df_tox['score_QED']>=qed_min) & (df_tox['SA_Score']<=sa_max)]
        if tox_only=="Non-toxiques ✅": dv = dv[dv['Tox_Prediction']==0]
        elif tox_only=="Toxiques ❌": dv = dv[dv['Tox_Prediction']==1]

        st.markdown(f"<p style='color: #94a3b8; font-style: italic;'>{len(dv)} molécules correspondent aux filtres</p>", unsafe_allow_html=True)

        mols = [Chem.MolFromSmiles(s) for s in dv.head(n_show)['smiles'] if Chem.MolFromSmiles(s)]
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250,200), legends=[f"QED: {r['score_QED']:.2f}" for i,r in dv.head(n_show).iterrows()])
            st.markdown('<div style="background: white; border-radius: 15px; padding: 10px;">', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.download_button("⬇️ Télécharger cette sélection", data=dv.to_csv(index=False).encode(), file_name='selection.csv')

    # --- ONGLET 3 : ESPACE LATENT ---
    with tab3:
        st.markdown('<div class="section-title">Visualisation de la Diversité Chimique</div>', unsafe_allow_html=True)
        try: st.image("umap_generated.png", use_container_width=True)
        except: st.info("Image UMAP manquante.")

    # --- ONGLET 4 : TOXICITÉ ML (XAI) ---
    with tab4:
        st.markdown('<div class="section-title">Predictor & Explainable AI (XAI)</div>', unsafe_allow_html=True)
        user_smi = st.text_input("🔬 Entrer un SMILES pour analyse temps-réel", "C1=CC=C(C=C1)[N+](=O)[O-]")

        if user_smi and best_model:
            mol = Chem.MolFromSmiles(user_smi)
            if mol:
                feat = extract_features(user_smi).reshape(1,-1)
                prob = best_model.predict_proba(feat)[0][1]

                c1, c2 = st.columns(2)
                with c1:
                    clr = "#ef4444" if prob > 0.5 else "#10b981"
                    st.markdown(f"""<div style='border:2px solid {clr}; border-radius:20px; padding:35px; text-align:center; background:{clr}11'>
                        <h1 style='color:{clr}; margin:0'>{'TOXIQUE' if prob > 0.5 else 'SÛR'}</h1>
                        <p style='font-size:1.4em; margin-top:10px;'>Probabilité : {prob*100:.1f}%</p>
                    </div>""", unsafe_allow_html=True)
                with c2: st.image(Draw.MolToImage(mol, size=(350,300)))

                st.markdown("---")
                # SHAP + HIGHLIGHT
                col_sh, col_at = st.columns(2)
                with col_sh:
                    st.markdown("### 🧠 Pourquoi ce score ? (SHAP)")
                    explainer = shap.TreeExplainer(best_model)
                    shap_v = explainer.shap_values(feat)
                    sv_plot = shap_v[1][0] if isinstance(shap_v, list) else shap_v[0,:,1] if len(shap_v.shape)==3 else shap_v[0]
                    idx = np.argsort(np.abs(sv_plot))[-10:]
                    fig, ax = plt.subplots(figsize=(8,5)); fig.patch.set_alpha(0); ax.set_facecolor('none')
                    ax.barh([FEATURE_NAMES[i] for i in idx], sv_plot[idx], color=['#ef4444' if x>0 else '#10b981' for x in sv_plot[idx]])
                    ax.tick_params(colors='white'); st.pyplot(fig)

                with col_at:
                    st.markdown("### 🔴 Atomes suspects")
                    info = {}; rdMolDescriptors.GetMorganFingerprint(mol, 2, bitInfo=info)
                    fp_pos = sorted([(i, sv_plot[i]) for i,n in enumerate(FEATURE_NAMES) if 'FP_bit_' in n and sv_plot[i]>0.01], key=lambda x: x[1], reverse=True)
                    h_atoms = set()
                    for bit_idx, _ in fp_pos[:5]:
                        b_id = int(FEATURE_NAMES[bit_idx].replace('FP_bit_',''))
                        if b_id in info:
                            for aidx, _ in info[b_id]: h_atoms.add(aidx)
                    drw = rdMolDraw2D.MolDraw2DSVG(450, 350)
                    rdMolDraw2D.PrepareMolForDrawing(mol)
                    drw.DrawMolecule(mol, highlightAtoms=list(h_atoms), highlightAtomColors={a:(0.9,0.2,0.2) for a in h_atoms})
                    drw.FinishDrawing()
                    st.markdown(f"<div style='background:white; border-radius:15px; padding:10px; text-align:center;'>{drw.GetDrawingText()}</div>", unsafe_allow_html=True)

                # --- COUNTERFACTUAL + CONCLUSION ---
                st.markdown("### 🟢 Solutions d'optimisation (Counterfactual)")
                results_cf = []
                for rs, desc in CF_MODIFICATIONS:
                    try:
                        rxn = AllChem.ReactionFromSmarts(rs)
                        prods = rxn.RunReactants((mol,))
                        if prods:
                            s_n = Chem.MolToSmiles(prods[0][0]); f_n = extract_features(s_n)
                            p_n = best_model.predict_proba([f_n])[0][1]
                            results_cf.append({"Modification": desc, "P(tox) Nouvelle": p_n * 100, "Impact": (p_n - prob) * 100})
                    except: continue

                if results_cf:
                    df_cf = pd.DataFrame(results_cf)
                    best_opt = df_cf.loc[df_cf['P(tox) Nouvelle'].idxmin()]
                    st.markdown(f"""
                        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 12px; padding: 20px; margin-bottom: 20px;'>
                            <h4 style='color: #10b981; margin: 0;'>🏆 Recommandation Stratégique</h4>
                            <p style='color: #e2e8f0; margin: 10px 0 0;'>L'analyse préconise la modification : <b>{best_opt['Modification']}</b>.</p>
                            <p style='color: #cbd5e1; font-size: 0.9em;'>Réduction de toxicité : <b>{best_opt['Impact']:.1f}%</b> | Probabilité finale : <b>{best_opt['P(tox) Nouvelle']:.1f}%</b>.</p>
                        </div>""", unsafe_allow_html=True)
                    st.table(df_cf.assign(Impact=lambda x: x['Impact'].map('{:+.1f}%'.format), **{'P(tox) Nouvelle': lambda x: x['P(tox) Nouvelle'].map('{:.1f}%'.format)}))

    # --- ONGLET 5 : DONNÉES ---
    with tab5:
        st.markdown('<div class="section-title">Registre complet des données</div>', unsafe_allow_html=True)
        st.dataframe(df_tox, use_container_width=True)

st.markdown("<p style='text-align:center; padding:40px; color:#4a5568;'>Hiba Mhada & Wijdane Bassiry • FSBM Casablanca • 2025</p>", unsafe_allow_html=True)
