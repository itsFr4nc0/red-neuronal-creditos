import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import gradio as gr

# ── Arquitectura (idéntica al notebook) ──────────────────────────────────────
class RedRiesgo(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.model(x)

# ── Carga de artefactos ───────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
preprocessor = joblib.load(os.path.join(BASE, "preprocessor.pkl"))
meta         = joblib.load(os.path.join(BASE, "meta.pkl"))
net          = RedRiesgo(meta["input_dim"])
net.load_state_dict(torch.load(os.path.join(BASE, "best_model.pt"),
                               map_location="cpu", weights_only=False))
net.eval()

# ── Helpers ───────────────────────────────────────────────────────────────────
sub_grade_map = {"A": "A3", "B": "B3", "C": "C3",
                 "D": "D3", "E": "E3", "F": "F3", "G": "G3"}

def band_info(score):
    if score >= 700: return "🟢 Muy Bajo Riesgo",  "#2e7d32"
    if score >= 600: return "🟡 Bajo Riesgo",       "#558b2f"
    if score >= 500: return "🟠 Riesgo Moderado",   "#e65100"
    if score >= 400: return "🔴 Alto Riesgo",        "#c62828"
    return "⛔ Muy Alto Riesgo", "#880e4f"

def calcular_score(
    loan_amnt, term, int_rate, purpose,
    annual_inc, home_ownership, emp_length,
    dti, revol_util, total_acc, open_acc,
    delinq_2yrs, inq_last_6mths, grade, verification_status
):
    r           = int_rate / 100 / 12
    installment = round(loan_amnt * r / (1 - (1 + r) ** (-term)), 2) if r > 0 else round(loan_amnt / term, 2)

    user_data = {
        "loan_amnt": float(loan_amnt), "funded_amnt": float(loan_amnt),
        "funded_amnt_inv": float(loan_amnt), "term": float(term),
        "int_rate": float(int_rate), "installment": float(installment),
        "emp_length": float(emp_length), "annual_inc": float(annual_inc),
        "dti": float(dti), "delinq_2yrs": float(delinq_2yrs),
        "inq_last_6mths": float(inq_last_6mths),
        "mths_since_last_delinq": 35.0, "mths_since_last_record": 0.0,
        "open_acc": float(open_acc), "pub_rec": 0.0, "revol_bal": 15000.0,
        "revol_util": float(revol_util), "total_acc": float(total_acc),
        "collections_12_mths_ex_med": 0.0, "mths_since_last_major_derog": 0.0,
        "policy_code": 1.0, "acc_now_delinq": 0.0, "tot_coll_amt": 0.0,
        "tot_cur_bal": 50000.0, "total_rev_hi_lim": 30000.0,
        "grade": grade, "sub_grade": sub_grade_map[grade],
        "home_ownership": home_ownership, "verification_status": verification_status,
        "pymnt_plan": "n", "purpose": purpose, "addr_state": "CA",
        "initial_list_status": "f", "application_type": "INDIVIDUAL",
    }

    df = pd.DataFrame([user_data])
    for col in meta["num_cols"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in meta["cat_cols"]:
        df[col] = df[col].astype(str)

    result = preprocessor.transform(df)
    X = result.toarray() if hasattr(result, "toarray") else np.array(result)

    with torch.no_grad():
        logit = net(torch.tensor(X, dtype=torch.float32)).item()

    p     = float(1 / (1 + np.exp(-logit)))
    p_c   = np.clip(p, 1e-6, 1 - 1e-6)
    score = int(np.clip(600 - 50 * np.log(p_c / (1 - p_c)), 300, 850))

    band, color = band_info(score)

    # Percentil aproximado (distribución empírica del test set)
    rng      = np.random.RandomState(42)
    pop      = np.clip(rng.normal(meta["score_mean"], meta["score_std"], 60_000), 300, 850).astype(int)
    pct      = float((pop < score).mean() * 100)

    resultado = f"""
## 💳 Tu Score Crediticio

<div style='text-align:center; font-size:96px; font-weight:900; color:{color}'>{score}</div>
<div style='text-align:center; font-size:28px'>{band}</div>
<div style='text-align:center; color:#555; margin-top:8px'>
  Probabilidad de incumplimiento: <b>{p*100:.1f}%</b><br>
  Cuota mensual estimada: <b>${installment:,.0f}</b><br>
  Estás en el percentil <b>{pct:.0f}</b> — mejor que el {pct:.0f}% de solicitantes
</div>

---
### 📋 Bandas de Riesgo
| Banda | Score | Tasa default |
|---|---|---|
| 🟢 Muy Bajo Riesgo | 700–850 | ~3% |
| 🟡 Bajo Riesgo | 600–699 | ~13% |
| 🟠 Riesgo Moderado | 500–599 | ~34% |
| 🔴 Alto Riesgo | 400–499 | ~77% |
| ⛔ Muy Alto Riesgo | 300–399 | >90% |
"""
    return resultado

# ── UI con Gradio Blocks ──────────────────────────────────────────────────────
with gr.Blocks(title="CreditScore AI 💳", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💳 CreditScore AI\nCalcula tu score crediticio con inteligencia artificial.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💰 Información del préstamo")
            loan_amnt  = gr.Slider(500, 40000, value=10000, step=500,  label="Monto solicitado (USD)")
            term       = gr.Radio([36, 60], value=36, label="Plazo (meses)")
            int_rate   = gr.Slider(5.0, 35.0, value=13.0, step=0.1,   label="Tasa de interés (%)")
            purpose    = gr.Dropdown(
                ["debt_consolidation", "credit_card", "home_improvement",
                 "other", "major_purchase", "small_business", "car",
                 "medical", "moving", "vacation", "house", "wedding",
                 "renewable_energy", "educational"],
                value="debt_consolidation", label="Propósito")

            gr.Markdown("### 👤 Información personal")
            annual_inc     = gr.Number(value=55000, label="Ingreso anual (USD)")
            home_ownership = gr.Dropdown(["RENT","MORTGAGE","OWN","OTHER"], value="RENT", label="Vivienda")
            emp_length     = gr.Slider(0, 10, value=5, step=1, label="Años de experiencia laboral")

            gr.Markdown("### 📊 Historial crediticio")
            dti        = gr.Slider(0.0, 100.0, value=15.0, step=0.5, label="Relación deuda/ingreso (DTI %)")
            revol_util = gr.Slider(0.0, 150.0, value=50.0, step=0.5, label="Utilización crédito rotativo (%)")
            total_acc  = gr.Number(value=22, label="Total cuentas de crédito")
            open_acc   = gr.Number(value=10, label="Cuentas abiertas")
            delinq_2yrs    = gr.Number(value=0, label="Mora (últimos 2 años)")
            inq_last_6mths = gr.Number(value=0, label="Consultas de crédito (últimos 6 meses)")
            grade              = gr.Dropdown(["A","B","C","D","E","F","G"], value="B", label="Grado crediticio")
            verification_status = gr.Dropdown(
                ["Not Verified","Source Verified","Verified"],
                value="Not Verified", label="Verificación de ingresos")

            btn = gr.Button("🔍 Calcular Score", variant="primary", size="lg")

        with gr.Column():
            resultado = gr.Markdown(
                value="## Ingresa tus datos y presiona **Calcular Score** →",
                elem_id="resultado"
            )

    btn.click(
        fn=calcular_score,
        inputs=[loan_amnt, term, int_rate, purpose,
                annual_inc, home_ownership, emp_length,
                dti, revol_util, total_acc, open_acc,
                delinq_2yrs, inq_last_6mths, grade, verification_status],
        outputs=resultado,
    )

    gr.Markdown(
        f"---\n🤖 Red Neuronal (256→128→64→1, BatchNorm) · "
        f"AUC-ROC: {meta['auc_nn']:.4f} · "
        f"[Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset)"
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, inbrowser=True)
