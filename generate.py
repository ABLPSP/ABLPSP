from equation_generator import *
# e_10=generator_equations_by_len(10)
exs=[]
for l in range(5,11):
	e=generator_equations_by_len(l)
	exs.append(e)
print(exs)
