from example_test import run

previous_G = {
    'r1' : ['r1','r2','r3'],
    'r2' : ['r1', 'r2','r4'],
    'r3' : ['r1', 'r3','r7'],
    'r4' : ['r2', 'r4','r8'],
    'r5' : ['r6', 'r9', 'c3'], # ['r5','r6', 'r9', 'c3'],
    'r6' : ['r5','r6','c3'],
    'r7' : ['r3', 'r7','c1'],
    'r8' : ['r4', 'r8','r9'],
    'r9' : ['r8','r9', 'r13'], # ['r5', 'r8','r9', 'r13'],
    'r10': ['r10','c3'],
    'r11': ['r11','c3'],
    'r12': ['r12','r22', 'outside'],
    'r13': ['r9', 'r13','r24'],
    'r14': ['r14','r24'],
    'r15': ['r15','c3'],
    'r16': ['r16','c3'],
    'r17': ['r17','c3'],
    'r18': ['r18','c3'],
    'r19': ['r19','c3'],
    'r20': ['r20','c3'],
    'r21': ['r21','c3'],
    'r22': ['r12', 'r22','r25'],
    'r23': ['r23','r24'],
    'r24': ['r13', 'r14', 'r23', 'r24'], # ['r13', 'r14', 'r23', 'r24'],
    'r25': ['r22', 'r25','r26'],
    'r26': ['r25', 'r26','r27'],
    'r27': ['r26', 'r27','r32'],
    'r28': ['r28','c4'],
    'r29': ['r29','r30', 'c4'],
    'r30': ['r29','r30'],
    'r31': ['r31','r32'],
    'r32': ['r27', 'r31','r32', 'r33'], # ['r27', 'r31','r32', 'r33'],
    'r33': ['r32','r33'],
    'r34': ['r34','c2'],
    'r35': ['r35','c4'],
    'c1' : ['r7', 'r25','c1', 'c2'],
    'c2' : ['r34', 'c1','c2', 'c4'],
    'c3' : ['c3'], # ['r5', 'r6', 'r10', 'r11', 'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'c3','o1'],
    'c4' : ['c4', 'r28', 'o1', 'c2'], # ['r28', 'r29', 'r35', 'c2','c4', 'o1'],
    'o1' : ['c3', 'c4','o1'],
    'outside': ['r12','outside']
}

print(run())

print(run())
print(run())
