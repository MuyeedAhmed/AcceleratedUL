import matplotlib.pyplot as plt

r1 = [2, 1]
r2 = [1, 2]
r3 = [6, 4]
r4 = [7, 3]
r5 = [2, 3]
r6 = [3, 2]

marker_size = 700

plt.figure(figsize=(6, 3.5))

# plt.scatter(r1[0], r1[1], color='blue', marker='s', s=marker_size)
# plt.scatter(r2[0], r2[1], color='blue', marker='s', s=marker_size)
# plt.scatter(r3[0], r3[1], color='red', marker='s', s=marker_size)
# plt.scatter(r4[0], r4[1], color='red', marker='o', s=marker_size)
# plt.scatter(r5[0], r5[1], color='blue', marker='o', s=marker_size)
# plt.scatter(r6[0], r6[1], color='red', marker='o', s=marker_size)

# plt.text(r1[0], r1[1], r'$R_1$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r2[0], r2[1], r'$R_2$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r3[0], r3[1], r'$R_3$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r4[0], r4[1], r'$R_4$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r5[0], r5[1], r'$R_5$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r6[0], r6[1], r'$R_6$', fontsize=14, ha='center', va='center', color='white')


# plt.scatter([], [], color='blue', marker='s', s=20, label=r'Sample $S_1$, Cluster: a')
# plt.scatter([], [], color='red', marker='s', s=20, label=r'Sample $S_1$, Cluster: b')
# plt.scatter([], [], color='blue', marker='o', s=20, label=r'Sample $S_2$, Cluster: a')
# plt.scatter([], [], color='red', marker='o', s=20, label=r'Sample $S_2$, Cluster: b')
# plt.legend()

# plt.xticks([])
# plt.yticks([])
# plt.legend()
# plt.xlim(0, 9)
# plt.ylim(0, 7)
# plt.savefig('Figures/MergeEx1.pdf', bbox_inches='tight')

# plt.show()



marker_size = 100

plt.figure(figsize=(6, 3.5))

plt.scatter(r1[0], r1[1], color='blue', marker='s', s=marker_size, alpha=0.2)
plt.scatter(r2[0], r2[1], color='blue', marker='s', s=marker_size, alpha=0.2)
plt.scatter(r3[0], r3[1], color='red', marker='s', s=marker_size, alpha=0.2)
plt.scatter(r4[0], r4[1], color='red', marker='o', s=marker_size, alpha=0.2)
plt.scatter(r5[0], r5[1], color='blue', marker='o', s=marker_size, alpha=0.2)
plt.scatter(r6[0], r6[1], color='red', marker='o', s=marker_size, alpha=0.2)

# plt.text(r1[0], r1[1], r'$R_1$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r2[0], r2[1], r'$R_2$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r3[0], r3[1], r'$R_3$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r4[0], r4[1], r'$R_4$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r5[0], r5[1], r'$R_5$', fontsize=14, ha='center', va='center', color='white')
# plt.text(r6[0], r6[1], r'$R_6$', fontsize=14, ha='center', va='center', color='white')


plt.scatter(1.5, 1.5, color='blue', marker='s', s=500, label='')
plt.scatter(r3[0], r3[1], color='red', marker='s', s=500,label='')
plt.scatter(r4[0], r4[1], color='blue', marker='o', s=500,label='')
plt.scatter(2.5, 2.5, color='red', marker='o', s=500,label='')

plt.scatter([], [], color='blue', marker='s', s=20, label=r'Cluster Center: a, $S_1$')
plt.scatter([], [], color='red', marker='s', s=20, label=r'Cluster Center: b, $S_1$')
plt.scatter([], [], color='blue', marker='o', s=20, label=r'Cluster Center: a, $S_2$')
plt.scatter([], [], color='red', marker='o', s=20, label=r'Cluster Center: b, $S_2$')
# plt.legend()


# plt.plot([1.5, r3[0], r4[0], 2.5, 1.5], [1.5, r3[1], r4[1], 2.5, 1.5], '--', label='Distance', color='black')
plt.plot([1.5, r3[0]], [1.5, r3[1]], '--', color='black', alpha=0.5)
plt.plot([r3[0], r4[0]], [r3[1], r4[1]], '--' , label='Euclidean Distance', color='black')
plt.plot([r4[0], 2.5], [r4[1], 2.5], '--', color='black', alpha=0.5)
plt.plot([2.5, 1.5], [2.5, 1.5], '--', color='black')

plt.plot([1.5, r4[0]], [1.5, r4[1]], '--', color='black', alpha=0.5)
plt.plot([2.5, r3[0]], [2.5, r3[1]], '--', color='black', alpha=0.5)


plt.xticks([])
plt.yticks([])
plt.legend()
plt.xlim(0, 9)
plt.ylim(0, 7)
plt.savefig('Figures/MergeEx2.pdf', bbox_inches='tight')

plt.show()


