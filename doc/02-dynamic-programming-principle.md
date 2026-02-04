# 动态规划算法 - 原理及适用条件

## 基本概念

**动态规划（Dynamic Programming，简称 DP）**是一种将复杂问题分解为更小子问题的算法思想，通过保存子问题的解来避免重复计算，从而提高效率。

### 核心特征

动态规划问题通常具有以下两个关键特征：

1. **重叠子问题（Overlapping Subproblems）**
   - 问题可以分解为多个子问题
   - 不同的子问题会包含相同的更小子问题
   - 保存子问题的解可以避免重复计算

2. **最优子结构（Optimal Substructure）**
   - 问题的最优解包含其子问题的最优解
   - 可以通过组合子问题的最优解构造原问题的最优解

### 设计步骤

解决动态规划问题通常遵循以下步骤：

1. **定义状态**：明确 dp 数组/状态变量代表什么含义
2. **建立状态转移方程**：找出状态之间的递推关系
3. **确定初始条件和边界**：设置 dp 数组的初始值
4. **确定计算顺序**：保证计算某状态时，其依赖的状态已计算
5. **确定答案**：从 dp 数组中获取最终答案

---

## 适用条件

### 1. 最优子结构性质

**问题**：问题的最优解是否可以由其子问题的最优解组合得到？

**判断方法**：
- 如果从原问题的最优解中提取一部分，得到的是子问题的最优解，则具有最优子结构
- 最短路径问题具有此性质（路径的一部分也是最短路径）
- 最长简单路径问题不具有此性质（路径的一部分未必是最长的）

**示例**：
```
最短路径问题：
A → B → C 的最短路径
= A → B 的最短路径 + B → C 的最短路径
```

### 2. 无后效性

**问题**：未来决策是否只依赖当前状态，与如何到达当前状态无关？

**含义**：
- 一旦某个状态确定了，它之后的发展只与当前状态有关
- 不需要关心"是如何到达这个状态的"
- 每个状态是"独立的"

**示例对比**：

| 具有无后效性 | 不具有无后效性 |
|-------------|---------------|
| 背包问题 | 某些图的最短路径（带负权边） |
| 最长递增子序列 | 博弈类问题（考虑对手决策历史） |
| 爬楼梯问题 | 需要记录路径的问题 |

### 3. 重叠子问题

**问题**：递归求解时是否有大量重复的子问题？

**判断方法**：
- 绘制递归树，观察是否有重复节点
- 如果子问题重复出现，可以用动态规划优化

**示例：斐波那契数列**

递归树：
```
                    f(5)
                /        \
             f(4)         f(3)
           /      \      /     \
        f(3)     f(2)  f(2)   f(1)
       /    \
     f(2)   f(1)
```

f(2)、f(3) 等子问题被重复计算多次 → 适合用动态规划

---

## 常见问题分类

### 1. 背包类

| 问题类型 | 特点 | 状态转移 |
|---------|------|---------|
| 01背包 | 每个物品选或不选 | `dp[j] = max(dp[j], dp[j-w] + v)` |
| 完全背包 | 每个物品可选多次 | `dp[j] = max(dp[j], dp[j-w] + v)` 正序遍历 |
| 多重背包 | 每个物品有限制次数 | 转化为01背包或二进制优化 |
| 二维费用背包 | 两个约束条件 | `dp[j][k] = max(dp[j][k], dp[j-w1][k-w2] + v)` |

### 2. 序列类

| 问题 | 特点 | 状态定义 |
|-----|------|---------|
| 最长递增子序列(LIS) | 单调子序列 | `dp[i]` = 以i结尾的最长递增子序列长度 |
| 最长公共子序列(LCS) | 两个序列 | `dp[i][j]` = 两个序列前i、j个字符的最长公共子序列 |
| 编辑距离 | 两个字符串转换 | `dp[i][j]` = 将前i个字符转换为前j个字符的最小操作数 |

### 3. 路径类

| 问题 | 特点 | 状态定义 |
|-----|------|---------|
| 最短路径 | 图中路径 | `dp[i]` = 从起点到i的最短距离 |
| 独特路径 | 网格中移动 | `dp[i][j]` = 从起点到(i,j)的路径数 |
| 最小路径和 | 网格和最小 | `dp[i][j]` = 从起点到(i,j)的最小路径和 |

### 4. 区间类

| 问题 | 特点 | 状态定义 |
|-----|------|---------|
| 区间DP | 子区间最优 | `dp[i][j]` = 区间[i,j]的最优解 |
| 回文子串 | 子串回文 | `dp[i][j]` = s[i..j]是否为回文 |
| 合并石子 | 区间合并 | `dp[i][j]` = 合并[i,j]区间的最小代价 |

### 5. 状态压缩类

- 集合表示、棋盘状态等
- 用整数表示状态
- 通常规模较小（如 n ≤ 20）

---

## 核心技巧

### 1. 状态定义的技巧

| 技巧 | 适用场景 | 示例 |
|-----|---------|------|
| 从最后一步倒推 | 序列、路径类 | "最后一个元素是..." |
| 从前向后定义 | 区间、累积类 | "前i个元素..." |
| 二维状态 | 两个约束/两个序列 | 背包、LCS |
| 多维状态 | 多个约束 | 三维背包、多维状态机 |

### 2. 空间优化技巧

```python
# 二维 → 一维优化
for i in range(n):
    for j in range(m, -1, -1):  # 01背包：倒序
        dp[j] = max(dp[j], dp[j-1] + value)
    # 或
    for j in range(m):  # 完全背包：正序
        dp[j] = max(dp[j], dp[j-1] + value)
```

### 3. 初始化技巧

| 场景 | 初始化方式 |
|-----|----------|
| 求最大值 | 初始化为 -∞ 或 0 |
| 求最小值 | 初始化为 ∞ 或 0 |
| 计数问题 | 初始化为 0 或 1 |
| 布尔问题 | 初始化为 false 或 true |

### 4. 遍历顺序

```python
# 一维数组：01背包（倒序）
for i in range(n):
    for j in range(W, weights[i]-1, -1):
        dp[j] = max(dp[j], dp[j-weights[i]] + values[i])

# 一维数组：完全背包（正序）
for i in range(n):
    for j in range(weights[i], W+1):
        dp[j] = max(dp[j], dp[j-weights[i]] + values[i])
```

---

## Python 示例代码

### 示例1：爬楼梯问题

```python
def climb_stairs(n):
    """
    爬楼梯问题：每次可以爬1或2个台阶，求n层楼梯有多少种爬法

    具有最优子结构：爬到第n层的方案数 = 爬到第n-1层的方案数 + 爬到第n-2层的方案数
    具有无后效性：只关心当前在第几层，不关心如何到达
    具有重叠子问题：递归时重复计算
    """
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


# 空间优化版
def climb_stairs_optimized(n):
    if n <= 2:
        return n

    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1
```

### 示例2：最长递增子序列

```python
def length_of_lis(nums):
    """
    最长递增子序列：找出数组中最长的严格递增子序列长度

    状态定义：dp[i] = 以nums[i]结尾的最长递增子序列长度
    状态转移：dp[i] = max(dp[j] + 1) for all j < i and nums[j] < nums[i]
    """
    n = len(nums)
    if n == 0:
        return 0

    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


# 优化版（二分查找）
def length_of_lis_optimized(nums):
    import bisect

    if not nums:
        return 0

    tails = []
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num

    return len(tails)
```

### 示例3：最长公共子序列

```python
def longest_common_subsequence(text1, text2):
    """
    最长公共子序列：两个字符串的最长公共子序列长度

    状态定义：dp[i][j] = text1前i个字符和text2前j个字符的LCS长度
    状态转移：
        - text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
        - 否则: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
```

### 示例4：最小路径和

```python
def min_path_sum(grid):
    """
    最小路径和：从左上到右下，只能向右或向下，求最小路径和

    状态定义：dp[i][j] = 从(0,0)到(i,j)的最小路径和
    状态转移：dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    """
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]

    # 初始化第一行和第一列
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]

    # 填充dp表
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])

    return dp[m - 1][n - 1]


# 空间优化版
def min_path_sum_optimized(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n

    dp[0] = grid[0][0]
    for j in range(1, n):
        dp[j] = dp[j - 1] + grid[0][j]

    for i in range(1, m):
        dp[0] = dp[0] + grid[i][0]
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j - 1])

    return dp[n - 1]
```

### 示例5：区间DP - 合并石子

```python
def min_cost_to_merge_stones(stones, K):
    """
    合并石子：每次合并K堆石头，代价为石头总数，求最小代价

    状态定义：dp[i][j] = 合并stones[i..j]为1堆的最小代价
    状态转移：dp[i][j] = min(dp[i][m] + dp[m+1][j]) + sum(i,j)
    """
    n = len(stones)
    if (n - 1) % (K - 1) != 0:
        return -1  # 无法合并

    # 前缀和
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + stones[i]

    # dp[i][j] 表示合并stones[i..j]的最小代价
    dp = [[0] * n for _ in range(n)]

    # 按长度递增
    for length in range(K, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            # 尝试不同的分割点
            for m in range(i, j, K - 1):
                dp[i][j] = min(dp[i][j], dp[i][m] + dp[m + 1][j])
            # 如果能合并成一堆，加上当前合并的代价
            if (length - 1) % (K - 1) == 0:
                dp[i][j] += prefix[j + 1] - prefix[i]

    return dp[0][n - 1]
```

---

## 判断是否适用动态规划

### 检查清单

| 检查项 | 问题 | 结果 |
|-------|------|------|
| 1 | 问题能否分解为子问题？ | 是/否 |
| 2 | 子问题是否有重叠（是否重复计算）？ | 是/否 |
| 3 | 最优解是否包含子问题的最优解？ | 是/否 |
| 4 | 状态是否满足无后效性？ | 是/否 |
| 5 | 数据规模是否适中（O(n²)、O(n³)可接受）？ | 是/否 |

如果以上检查项大部分为"是"，则问题适合用动态规划解决。

### 与其他算法的对比

| 算法 | 特点 | 适用场景 |
|-----|------|---------|
| 动态规划 | 重叠子问题，保存结果 | 最优化问题、计数问题 |
| 贪心 | 每步选局部最优 | 某些最优化问题 |
| 分治 | 子问题独立，不重叠 | 排序、归并等 |
| 回溯 | 暴力搜索所有可能 | 组合、排列问题 |

---

## 常见错误与注意事项

### 1. 状态定义错误

```python
# 错误：状态定义不清楚
def wrong_example(nums):
    dp = [0] * len(nums)
    # dp[i]的含义不明确...

# 正确：明确状态含义
def right_example(nums):
    # dp[i] = 以nums[i]结尾的最长递增子序列长度
    dp = [1] * len(nums)
```

### 2. 边界条件处理

```python
# 注意边界情况
def climb_stairs(n):
    if n <= 2:
        return n  # 处理小规模
    # ...
```

### 3. 遍历顺序错误

```python
# 01背包必须倒序遍历
for i in range(n):
    for j in range(W, weights[i]-1, -1):  # 倒序
        dp[j] = max(dp[j], dp[j-weights[i]] + values[i])
```

### 4. 空间过度优化

```python
# 过度优化可能导致代码难以理解和调试
# 在不必要的情况下，保持代码清晰更重要
```

---

## 复杂度分析

### 一般复杂度

| 问题类型 | 时间复杂度 | 空间复杂度 |
|---------|-----------|-----------|
| 一维线性DP | O(n) 或 O(nm) | O(n) 或 O(m) |
| 二维DP | O(n²) 或 O(nm) | O(n²) 或 O(nm) |
| 三维DP | O(n³) | O(n³) 或 O(n²) |
| 区间DP | O(n³) | O(n²) |

### 优化技巧

1. **空间优化**：根据依赖关系减少维度
2. **单调性优化**：利用数据单调性减少内层循环
3. **数据结构优化**：用线段树、树状数组等加速查询
4. **斜率优化 / 四边形不等式优化**：针对特定DP的数学优化

---

## 练习建议

1. **基础题目**
   - 爬楼梯、斐波那契数列
   - 爬楼梯的最小代价
   - 打家劫舍

2. **进阶题目**
   - 背包问题系列
   - LIS、LCS
   - 最小路径和、不同路径

3. **高级题目**
   - 区间DP（石子合并、回文子串）
   - 状态压缩DP
   - 数位DP

---

## 参考资料

- 《算法导论》第15章：动态规划
- 《算法竞赛进阶指南》
- LeetCode 动态规划标签
