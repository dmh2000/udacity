# ----------------------
# P(Jill Stein)
p_j = 0.5

# P(freedom | J)
p_fj = 0.1

# P(immigration | J)
p_ij = 0.1

# P(environment | J)
p_ej = 0.8

p_j_text = p_j * p_fj * p_ij

print(p_j_text)

# ----------------------
# P(Gary Johnson)
p_g = 0.5

# P(freedom | G)
p_fg = 0.7

# P(immigration | G)
p_ig = 0.2

# P(environment | J)
p_eg = 0.1

p_g_text = p_g * p_fg * p_ig

print(p_g_text)

# ------------------------
p_f_i = p_j_text + p_g_text
print(p_f_i)

# P(J|F,I) = (P(J) * P(F|J) * P(I|J)) / P(F,I)
p_j_fi = (p_j * p_fj * p_ij) / p_f_i
print('jill stein says freedom and immigration ', p_j_fi)

# P(G | F,I) = (P(G) * P(F|G) * P(I|G)) / P(F,I)
p_g_fi = (p_g * p_fg * p_ig) / p_f_i
print('gary johnson says freedom and immigration',p_g_fi)