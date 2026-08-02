# Appendix B: GEAR Bilevel Search

**Algorithm 1: GEAR Bilevel Software--Hardware Search**

```text
Input:  Θ_sw, D_out^0, R, B, ρ, C_in
Output: P*_ρ

 1: (E_AP^(0), E_ℓ^(0), E_e^(0)) ← FITOUTER(D_out^0)
 2: for τ = 1, ..., R do
 3:     φ(x) ← [φ_cfg(x), φ_graph(x)]
 4:     f̂_r,τ(x) ← E_r^(τ−1)(φ(x)),  r ∈ {AP, ℓ, e}
 5:     F̂_τ(x) ← (−f̂_AP,τ(x), f̂_ℓ,τ(x), f̂_e,τ(x))
 6:     U_τ ← NSGA-II(Θ_sw, F̂_τ)
 7:     X_τ ← BATCH_B(U_τ, D_out^(τ−1))
 8:     ΔD_out^τ ← ∅
 9:     for x ∈ X_τ do
10:         Θ_hw(x;ρ) ← INSTANTIATE(x,ρ)
11:         D_in^x ← SEED(Θ_hw(x;ρ))
12:         while ¬C_in(D_in^x) do
13:             G_x ← FITXGB(D_in^x)
14:             Q_x ← SA(Θ_hw(x;ρ), G_x)
15:             Z_x ← BUILDMEASURE(x,Q_x,ρ)
16:             D_in^x ← D_in^x ∪ Z_x
17:         end while
18:         s_x* ← arg min_(s,ℓ_s)∈D_in^x ℓ_s
19:         y_x ← EVALUATE_ρ(x,s_x*)
20:         ΔD_out^τ ← ΔD_out^τ ∪ {(x,s_x*,y_x)}
21:     end for
22:     D_out^τ ← D_out^(τ−1) ∪ ΔD_out^τ
23:     (E_AP^(τ), E_ℓ^(τ), E_e^(τ)) ← FITOUTER(D_out^τ)
24: end for
25: P*_ρ ← PARETO_(−AP,ℓ,e)({d ∈ D_out^R | g_ρ(c_d) ∈ C_ρ})
26: return P*_ρ
```

`FITOUTER` returns the LightGBM AP predictor and the two ExtraTrees predictors. `BATCH` applies nondominated rank and crowding distance after excluding measured candidates, while `EVALUATE_ρ(x,s)=(AP(x),ℓ(x,s),e(x,s))` contains only target-profile measurements.
