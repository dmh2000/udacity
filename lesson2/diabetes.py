# P(D) is probability of someone having diabetes = 0.01
# P(Pos) is probability of a positive test result
# P(Neg) is probability of a negative test result
# P(Pos | D) is probability of a positive result given a person has diabetes = 0.9
#   Sensitivity
# P(Neg | ~D) is probability of a negative result given a person does not have diabetes = 0.9
#   Specificity

# bayes = P(A|B) = P(B|A) * P(A)
#                  -------------
#                      P(B)

# probability of A given B if we know B given A

# P(D | Pos) = P(Pos | D) * P(D)
#              ----------------
#                  P(Pos)

# get P(Pos) (any positive result)
# P(Pos) = (P(Pos | D) * P(D)) / P(D | Pos)
# P(Pos) = [P(D) * Sensitivity] + [P(~D) * (1 - Specificity)]

# P(D)
p_diabetes = 0.01

# P(~D)
p_no_diabetes = 0.99

# Sensitivity P(Pos | D)
p_pos_diabetes = 0.9

# Specificity P(Neg | ~D)
p_neg_no_diabetes = 0.9

# P(Pos)
p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1.0 - p_neg_no_diabetes))

print(p_pos)


# P(D|Pos) = (P(D) * Sensitivity) / P(Pos)
p_diabetes_pos = (p_diabetes * p_pos_diabetes) / p_pos

# P(~D/Pos) = (P(~D) * (1-Specificity)) / P(Pos)
p_no_diabetes_pos = (p_no_diabetes * (1-p_neg_no_diabetes)) / p_pos
print(p_diabetes_pos)
print(p_no_diabetes_pos)


