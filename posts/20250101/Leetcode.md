**日期：**2025-6-12

**题目：**给定一个整数数组 nums 和一个整数目标值 target，请在数组中找出和为目标值的那两个整数，并返回数组的下标，结果只有一个。

**思路：**一个正常简单的思路是使用两个指针遍历该数组，一个快指针，一个慢指针，然后进行相加判断，正确就结束遍历返回两个指针下标，显然这种时间复杂度为平方级别，如果数组过大的话，效率会很低。是否可以采用哈希表？哈希表的插入和查找的时间负责度都是常数级别。可以使用哈希表来存储已经遍历过的数字及其索引，这样可以在常数时间内查找是否存在另一个数字，使得当前数字与它的和等于 target，这样算法的复杂度就可以降到线性级别。

**代码：**

```python
# 暴力解法
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype List[int]
        """
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []

# 哈希表解法
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype List[int]
        """
        num_map = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_map:
                return [num_map[complement], i]
            num_map[num] = i
        return []
```

**日期：**2025-6-13

**题目：**给定两个非空的链表，表示两个非负的整数。它们每位数都是按照逆序的方式存储的，并且每个节点只能存储一位数字。请将两个数相加，并以相同的方式返回一个表示和的链表，可以假设除了数字0外，这两个数都不会以0开头。

**思路：**两个逆序的链表进行相加，然后还要以逆序的方式返回，由于数字是逆序存储的，所以可以直接从链表的头部相加，这当然遵循加法法则，无需反转链表，但是要注意进位的问题，在每一步的计算中，必须正确处理进位，并将其传递到下一步，还有就是如果两个链表的长度不相等，短的链表的在遍历结束后应该视为0 ，在所有的节点遍历完成后，如果还有进位，那么就应该将其作为新的节点添加到链表中。

**代码：**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
 
class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy_head = ListNode(0)	# 创建一个虚拟头节点
        current = dummy_head		# 用于构建结果链表
        carry = 0					# 用于存储进位
        
        while l1 or l2 or carry:
            val1 = (l1.val if l1 else 0)	# 如果链表已经遍历完，那么就视为0
            val2 = (l2.val if l2 else 0)
            
            total = val1 + val2 + carry
            carry = total // 10
            current_val = total % 10
            
            current.next = ListNode(current_val)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
                
        return dummy_head.next
```

**日期：**2025-6-18

**题目：**给定一个字符串 s，找出其中不含重复字符的最长子串的长度。

**思路：**这个显然有一个暴力解法，对于每一个可能的子串，检查它是否包含重复字符。