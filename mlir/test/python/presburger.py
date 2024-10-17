from mlir import presburger
import numpy as np

"""
Test the following integer relation

x + 2y = 8
x - y <= 1
y >= 3
"""
eqs = np.asarray([[1, 2, -8]], dtype=np.int64)
ineqs = np.asarray([[1, -1, -1], [0, -1, 3]], dtype=np.int64)
relation = presburger.IntegerRelation(ineqs, eqs, 2, 0)
print(relation)
print(relation.num_constraints)
print(relation.num_inequalities)
print(relation.num_equalities)
print(relation.num_domain_vars)
print(relation.num_range_vars)
print(relation.num_symbol_vars)
print(relation.num_local_vars)
print(relation.num_columns)

eq_first_row = relation.get_equality(0)
print(eq_first_row)
ineq_second_row = relation.get_inequality(1)
print(ineq_second_row)

eq_coefficients = relation.equalities()
print(eq_coefficients[0, 1])
ineq_coefficients = relation.inequalities()
print(ineq_coefficients[1, 1])

"""
Test intersection

Relation A

x + y <= 6

Relation B

x>=2
"""
eqs_a = np.asarray([[0, 0, 0]], dtype=np.int64)
ineqs_a = np.asarray([[-1, -1, 6]], dtype=np.int64)
relation_a = presburger.IntegerRelation(ineqs_a, eqs_a, 2, 0)
print(relation_a)

eqs_b = np.asarray([[0, 0, 0]], dtype=np.int64)
ineqs_b = np.asarray([[1, 0, -2]], dtype=np.int64)
relation_b = presburger.IntegerRelation(ineqs_b, eqs_b, 2, 0)
print(relation_b)

a_b_intersection = relation_a.intersect(relation_b)
print(a_b_intersection)
