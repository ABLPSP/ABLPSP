import sys
import time
import copy
import math
import numpy as np
sys.path.insert(0, '../lib/')
import heapq
import LogicLayer as LL
from zoopt import Dimension, Objective, Parameter, Opt, Solution

total_time=0
# logiclayer feature
class LL_feature:
	mapping = {}
	rules = []

	def __init__(self, m, r):
		self.mapping = m
		self.rules = r

	# evaluate ONE example
	def eval(self, ex):
		# apply mapping
		ex_symbs_n = apply_mapping(ex, self.mapping)
		(ex_symbs, n_pos) = remove_nulls(ex_symbs_n)
		ex_term = LL.PlTerm(ex_symbs)  # LL term
		return LL.evalInstFeature(ex_term, LL.PlTerm(self.rules))


# class of consistent result score
class consist_result:
	score = 0
	consistent_ex_ids = []
	abduced_exs = []  # {0,1,2,3,4,...} values for NN
	abduced_map = {}  # mapping of NN output to symbols
	abduced_exs_mapped = []  # Prolog rules

	def __init__(self):
		pass

	def __init__(self, max_score, max_exs, max_map, max_rules, indices):
		self.score = max_score
		self.abduced_exs = max_exs
		self.abduced_map = max_map
		self.abduced_exs_mapped = max_rules
		self.consistent_ex_ids = indices

	def copy(self):
		return consist_result(self.score, self.abduced_exs.copy(),
							  self.abduced_map.copy(),
							  self.abduced_exs_mapped.copy(),
							  self.consistent_ex_ids.copy())

	def to_feature(self):
		feat = LL.conInstsFeature(self.abduced_exs_mapped)
		return LL_feature(self.abduced_map, feat)


def find(vec):  # find first non zero index, for reading one-hot vector
	for i in range(0, len(vec)):
		if (vec[i] > 0):
			return i
	return -1


def gen_mappings(chars, symbs):
	n_char = len(chars)
	n_symbs = len(symbs)
	if n_char != n_symbs:
		print('Characters and symbols size dosen\'t match.')
		return
	from itertools import permutations
	mappings = []
	# returned mappings
	perms = permutations(symbs)
	for p in perms:
		mappings.append(dict(zip(chars, list(p))))
	return mappings


def apply_mapping(chars, mapping):
	re = []
	for c in chars:
		if c == '_':  # leave vars unchanged
			re.append(c)
		elif not (c in mapping):
			print('Wrong character for mapping.')
			return
		else:
			re.append(mapping[c])
	return re


def remove_nulls(chars):
	re = []
	null_pos = []
	for i in range(0, len(chars)):
		if not chars[i] == 'null':
			re.append(chars[i])
		else:
			null_pos.append(i)
	return (re, null_pos)


def add_nulls(chars, null_pos):
	for idx in null_pos:
		chars.insert(idx, 'null')
	return chars


def flatten(l):
	return [item for sublist in l for item in sublist]


# reformulate identifiers from flat list to list of lists as examples
def reform_ids(exs, var_ids):
	exs_f = flatten(exs)
	assert len(exs_f) == len(var_ids)
	re = []
	i = 0
	for e in exs:
		j = 0
		ids = []
		while j < len(e):
			ids.append(var_ids[i + j])
			j += 1
		re.append(ids)
		i = i + j
	return re #【【4个0】，【5个0】，【6个0】】


# substitute const in examples to vars according to the var identifiers (flatten)
def sub_vars(exs, var_ids_f):
	flag = False
	if len(exs) == len(var_ids_f):
		for i in range(0, len(exs)):
			if not len(exs[i]) == len(var_ids_f[i]):
				break
		flag = True
	if not flag:
		var_ids = reform_ids(exs, var_ids_f)
	else:
		var_ids = var_ids_f
	subd_exs = []
	for i in range(0, len(exs)):
		ex = exs[i]
		var_id = var_ids[i]
		assert len(ex) == len(var_id)
		subd = []
		for j in range(0, len(ex)):
			if var_id[j]:
				subd.append('_')  # replace a variable
			else:
				subd.append(ex[j])  # use original term
		subd_exs.append(subd)
	return subd_exs


# return a consistent score (number maximum consistent examples) of given list of examples and variable indicators
def consistent_score(exs, var_ids, maps):
	# for debug
	#print('vars: ', end = '\t')
	#print(var_ids)

	max_score = 0
	max_exs = []
	max_map = {}
	max_rules = []
	max_subd_exs_ids = []
	max_null_pos = []

	subd_exs_all = sub_vars(exs, var_ids)  # examples been replaced variables
	subd_exs_ids = []
	#count = 0
	t = time.time()
	for i in range(0, len(subd_exs_all)):
		subd_exs_ids.append(i)

		subd_exs = []
		for j in subd_exs_ids:
			subd_exs.append(subd_exs_all[j])

		got_con_sol = False
		# do mapping and evaluation, TODO: possible for multithread
		for m in maps:
			#LL.gc()
			#LL.trimStacks() # IMPORTANT!!
			mapped_subd_exs = []  # list of plterms

			null_pos = []
			for e in subd_exs:
				e_symbs_n = apply_mapping(e, m)
				(e_symbs, n_pos) = remove_nulls(e_symbs_n)
				null_pos.append(n_pos)
				mapped_subd_exs.append(LL.PlTerm(e_symbs))
			mapped_subd_term = LL.PlTerm(mapped_subd_exs)
			con_sol = LL.abduceConInsts(
				mapped_subd_term)  # consistent solution

			if con_sol:
				got_con_sol = True
			if con_sol and max_score < len(subd_exs):
				max_rules = copy.deepcopy(con_sol.py())
				max_score = len(subd_exs)
				max_subd_exs_ids = subd_exs_ids.copy()
				max_map = m.copy()
				max_null_pos = null_pos.copy()

				if max_score == len(mapped_subd_exs):
					break
		if not got_con_sol:
			subd_exs_ids.pop()

	abduced_exs = exs.copy()
	inv_m = {v: k for k, v in max_map.items()}  # inverted map
	#print(max_subd_exs_ids)
	for j in range(0, len(max_subd_exs_ids)):
		# add nulls back
		ex_n = add_nulls(max_rules[j].copy(), max_null_pos[j])
		mapped_ex = apply_mapping(ex_n, inv_m)
		abduced_exs[max_subd_exs_ids[j]] = mapped_ex
	'''
	print('elapsed time: ', end = '\t')
	print(time.time() - t, end = '\t')
	print('score:', end = '\t')
	print(max_score)
	'''
	re = consist_result(max_score, abduced_exs, max_map, max_rules,
						max_subd_exs_ids)
	return re


# this score evaluation does not iterate on mappings
def consistent_score_mapped(exs, var_ids, m):
	max_score = 0
	max_exs = []
	max_rules = []
	subd_exs = sub_vars(exs, var_ids)  # examples been replaced variables

	mapped_subd_exs = []  # list of plterms
	inv_m = {v: k for k, v in m.items()}  # inverted map
	null_pos = []
	for e in subd_exs:
		e_symbs_n = apply_mapping(e, m)
		(e_symbs, n_pos) = remove_nulls(e_symbs_n)
		null_pos.append(n_pos)
		mapped_subd_exs.append(LL.PlTerm(e_symbs))
	mapped_subd_term = LL.PlTerm(mapped_subd_exs)
	# print('????')
	# print('mapped_subd_term:',mapped_subd_term)
	con_sol = LL.abduceConInsts(mapped_subd_term)  # consistent solution 求解变量，符号为_ con_sol是规则
	# print('!!!!')
	if con_sol:
		max_rules = copy.deepcopy(con_sol.py())
		max_subd_exs = con_sol.py().copy()
		max_exs = []
		for k in range(0, len(max_subd_exs)):
			# add nulls back
			max_subd_exs_n = add_nulls(max_subd_exs[k], null_pos[k]) 
			# map back
			max_exs.append(apply_mapping(max_subd_exs_n, inv_m))# 映射为01+=
		abduced_exs = exs.copy()
		for i in range(0, len(max_exs)):
			abduced_exs[i] = max_exs[i]
		max_score = len(max_exs) # 满足规则的式子数量 abduced_exs满足规则的式子
		re = consist_result(max_score, abduced_exs, m, max_rules, [])
		return re
	else:
		return None


# this one does not iterate on mappings and return a set of equation sets
def consistent_score_sets(exs, var_ids_flat, mapping):
	global total_time
	s_time=time.time()
	var_ids = reform_ids(exs, var_ids_flat)
	lefted_ids = [i for i in range(0, len(exs))]
	consistent_set_size = []
	consistent_res = []
	# find consistent sets
	while lefted_ids:
		temp_ids = []
		temp_ids.append(lefted_ids.pop(0))
		max_con_ids = []
		max_con_res = None
		found = False
		for i in range(-1, len(exs)):
			if (not i in temp_ids) and (i >= 0):
				temp_ids.append(i)
			# test if consistent
			temp_exs = []
			temp_var_ids = []
			for i in temp_ids:
				temp_exs.append(exs[i])
				temp_var_ids.append(var_ids[i])
			# print('mapped')
			con_res = consistent_score_mapped(temp_exs, temp_var_ids, mapping)
			if not con_res:
				if len(temp_ids) > 1:
					temp_ids.pop()
			else:
				if len(temp_ids) > len(max_con_ids):
					found = True
					max_con_ids = temp_ids.copy()
					max_con_res = con_res.copy()
					'''
					print('con:', end = '\t')
					print(temp_ids)
					print(max_con_res.abduced_exs_mapped)
					print('left:', end = '\t')
					print([i for i in lefted_ids if i not in max_con_ids])
					'''

		removed = [i for i in lefted_ids if i in max_con_ids]

		if found:
			#input('Hit any key to continue')
			max_con_res.consistent_ex_ids = max_con_ids.copy() # 一个分组的下标
			consistent_res.append(max_con_res.copy())
			consistent_set_size.append(len(removed) + 1) # 每一分组的长度
			lefted_ids = [i for i in lefted_ids if i not in max_con_ids]

	consistent_set_size.sort()
	score = 0
	for i in range(0, len(consistent_set_size)):
		score += math.exp(-i) * consistent_set_size[i]
	e_time=time.time()
	total_time+=e_time-s_time
	return (score, consistent_res)


# optimise the variable indicators to find the best consistent abduction of examples
def opt_var_ids(exs, maps):
	dim = Dimension(len(flatten(exs)), [[0, 1]] * len(flatten(exs)), [False] * len(flatten(exs)))
	obj = Objective(lambda v: -consistent_score(exs, v.get_x(), maps).score, dim)
	param = Parameter(budget=100, autoset=True)
	solution = Opt.min(obj, param)

	return solution

def get_score(hard_label,mask_probability):
	score=1
	for i in range(len(hard_label)):
		score*=mask_probability[i][hard_label[i]]
	return score
class Heap:
    def __init__(self):
        self._heap = []

    def push(self, item):
        heapq.heappush(self._heap, item)

    def pop(self):
        return heapq.heappop(self._heap)

def sort_mask(mask_probability,budget=10):
	# budget=min(budget,int(math.pow(2,len(mask_probability))))
	selected_dict = {}
	mask_probability=np.array(mask_probability)
	# print('mask_probability:',mask_probability.shape)
	max_hard_label=np.argmax(mask_probability,axis=1)
	# print('max_hard_label:',max_hard_label)
	# print('max_hard_label:',max_hard_label)
	max_score=1
	for i in range(len(max_hard_label)):
		max_score*=mask_probability[i][max_hard_label[i]]
	# max_score=get_score(max_hard_label,mask_probability)
	V=[]
	for i in range(len(max_hard_label)):
		V.append(mask_probability[i][1-max_hard_label[i]]/mask_probability[i][max_hard_label[i]])
	V_list = list(enumerate(V))
	# print(V)
	# print("原始列表及索引:", indexed_list)

	# 对列表进行排序，以值为基准
	sorted_V_tuple = sorted(V_list, reverse=True, key=lambda x: x[1])
	# print("排序后的列表:", sorted_list)
	# print(sorted_V_tuple)
	# 获取排序后原先的位置
	original_positions = [x[0] for x in sorted_V_tuple]
	# print(original_positions)
	# sorted_V = [x[1] for x in sorted_V_tuple]
	# sorted_mask_probability=[mask_probability[i] for i in original_positions]
	origin_state=[copy.deepcopy(original_positions)]
	state=[copy.deepcopy(original_positions)]
	# print("排序后原先的位置:", original_positions)
	suc_prob=V[state[0][0]]*max_score
	# print(state[0][0])
	# print(V[state[0][0]])
	suc_mask=copy.deepcopy(max_hard_label)
	suc_mask[state[0][0]]=1-suc_mask[state[0][0]]
	max_heap = Heap()
	max_heap.push((-suc_prob,1))
	# print('max_score:',max_score)
	masks=[]
	prob=[]
	masks.append(max_hard_label)
	prob.append(max_score)
	selected_dict[tuple(max_hard_label)]=max_score
	selected_dict[tuple(suc_mask)]=suc_prob
	cur_budget=1
	sum_conflict=0
	max_conflict=0
	max_son_conflict=0
	max_father_conflict=0
	while cur_budget<budget:
		heaptop=max_heap.pop()
		suc_prob=-heaptop[0]
		father_idx=heaptop[1]
		origin_state_f=origin_state[father_idx-1]
		state_f=state[father_idx-1]
		mask_f=masks[father_idx-1]
		prob_f=prob[father_idx-1]
		# suc_mask=None
		# if len(state_f) is not 0:
		suc_mask=copy.deepcopy(mask_f)
		pos=state_f.pop(0)
		suc_mask[pos]=1-suc_mask[pos]
		# if tuple(next_mask) in selected_dict:
		# 	suc_mask=None
		# 	pos=None
	 	# else:
		origin_suc_state=copy.deepcopy(origin_state_f)
		origin_suc_state.remove(pos)
		origin_state.append(origin_suc_state)
		suc_state=copy.deepcopy(origin_suc_state)
		
		masks.append(suc_mask)
		prob.append(suc_prob)
		conflict=0
		while len(suc_state) is not 0:
			suc_suc_mask=copy.deepcopy(suc_mask)
			_pos=suc_state[0]
			suc_suc_mask[_pos]=1-suc_suc_mask[_pos]
			suc_suc_prob=suc_prob*V[_pos]
			if tuple(suc_suc_mask) in selected_dict:
				conflict+=1
				sum_conflict+=1
				# print('son')
				# print('conflict:',conflict)
				max_conflict=max(conflict,max_conflict)
				max_son_conflict=max(conflict,max_son_conflict)
				suc_state.pop(0)
				continue
			else:
				max_heap.push((-suc_suc_prob,cur_budget+1))
				selected_dict[tuple(suc_suc_mask)]=suc_suc_prob
				break
		state.append(suc_state)
		conflict=0
		while len(state_f) is not 0:
			suc_f_mask=copy.deepcopy(mask_f)
			_pos=state_f[0]
			suc_f_mask[_pos]=1-suc_f_mask[_pos]
			suc_f_prob=prob_f*V[_pos]
			if tuple(suc_f_mask) in selected_dict:
				state_f.pop(0)
				sum_conflict+=1
				conflict+=1
				# print('father')
				# print('conflict:',conflict)
				max_conflict=max(conflict,max_conflict)
				max_father_conflict=max(conflict,max_father_conflict)
				continue
			else:
				max_heap.push((-suc_f_prob,father_idx))
				selected_dict[tuple(suc_f_mask)]=suc_f_prob
				break
		state[father_idx-1]=state_f

 		
	 		# break



		# global_max_next_score=float('-inf')
		# global_max_next_mask=None
		# for mask in masks:
		# 	score=selected_dict[tuple(mask)]
		# 	max_next_score=float('-inf')
		# 	max_next_mask=None
		# 	for i in range(len(mask)):
		# 		if mask_probability[i][mask[i]]<mask_probability[i][1-mask[i]]:
		# 			continue
		# 		next_mask=copy.deepcopy(mask)
		# 		next_mask[i]=1-mask[i]
		# 		if tuple(next_mask) in selected_dict:
		# 			continue
		# 		next_score=score*mask_probability[i][1-mask[i]]/mask_probability[i][mask[i]]
		# 		if next_score>max_next_score:
		# 			max_next_score=next_score
		# 			max_next_mask=next_mask

		# 	if max_next_score>global_max_next_score:
		# 		global_max_next_mask=max_next_mask
		# 		global_max_next_score=max_next_score
		# masks.append(global_max_next_mask)
		# selected_dict[tuple(global_max_next_mask)]=global_max_next_score
		cur_budget+=1
	print(budget)
	print('sum_conflict:',sum_conflict)
	print('max_conflict:',max_conflict)
	print('max_father_conflict:',max_father_conflict)
	print('max_son_conflict:',max_son_conflict)
	# print('masks:',masks)
	# print('prob:',prob)
	print(np.sum(prob))
	return masks

# optimise the variable indicators to find the best consistent abduction of examples
def encoder_decoder_constraint(encoder_decoder_model, exs, probability, mapping, budget=10,max_length=8,input_size=4,T=1):
	mask_probability=[]
	for prob in probability:
		len_mask=len(prob)
		print('max_length:',max_length)
		print('len_mask:',len_mask)
		prob_padded = np.zeros((max_length-len_mask,input_size))
		prob = np.concatenate([prob_padded,prob])
		mask=encoder_decoder_model.predict(np.expand_dims(prob, axis=0), verbose=0)
		# print('prob:',prob)
		# print('mask:',mask)

		mask=np.squeeze(mask, axis=0)[max_length-len_mask:]
		# print('mask:',mask.shape)
		# print(np.sum(np.power(mask,0,5),axis=1,keepdims=True).shape)
		mask_probability.append(mask/T/np.sum(mask/T,axis=1,keepdims=True))

	mask_probability_cat = np.concatenate(mask_probability)
	# print('mask_probability_cat:',mask_probability_cat)

	masks = sort_mask(mask_probability_cat,budget)
	# print('masks:',masks)
	max_score = float('-inf')
	max_mask = None
	# i=0
	global total_time
	total_time=0
	for mask in masks:
		# print('i:',i)
		# print('mask:',mask)
		# print('mask:',mask)
		score = consistent_score_sets(exs, mask, mapping)[0]
		# print('score:',score)
		# i+=1
		# mask_ans=[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
		# if len(mask_ans)==len(mask):
		# 	score_ans = consistent_score_sets(exs, mask_ans, mapping)[0]
			# print('score_ans:',score_ans)
		if score > max_score:
			max_score = score
			max_mask = mask
	# print('max_mask',max_mask)
	# print('masks:',masks)
	# print('max_mask:',max_mask)
	solution=max_mask#np.random.randint(2,size=len(max_mask))#max_mask
	return solution,total_time
