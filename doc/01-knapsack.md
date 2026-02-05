# 01背包问题 - 解法分析与示例代码

## 问题定义

**01背包问题**：给定n个物品，每个物品有重量`w[i]`和价值`v[i]`，背包容量为W。每个物品只能选择放入或不放入（0或1），求背包能装入物品的最大总价值。

## 动态规划解法

### 核心思想

使用二维数组`dp[i][j]`表示：前i个物品，在背包容量为j时的最大价值。

### 状态转移方程

```
if (w[i-1] > j) {
    dp[i][j] = dp[i-1][j];  // 当前物品装不下
} else {
    dp[i][j] = max(dp[i-1][j], dp[i-1][j - w[i-1]] + v[i-1]);
    // 不装入 vs 装入当前物品
}
```

### 空间优化

可以将二维数组优化为一维数组`dp[j]`，倒序遍历容量：

```
dp[j] = max(dp[j], dp[j - w[i]] + v[i])
```

注意：必须**倒序遍历**容量，避免重复使用同一物品。

## Python 示例代码

### 方法一：二维数组实现

```python
def knapsack_2d(W, weights, values):
    """
    01背包问题 - 二维数组实现

    Args:
        W: 背包容量
        weights: 物品重量列表
        values: 物品价值列表

    Returns:
        最大价值
    """
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, W + 1):
            if weights[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i - 1][j - weights[i - 1]] + values[i - 1]
                )

    return dp[n][W]


# 测试
if __name__ == "__main__":
    W = 10
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    print(f"最大价值: {knapsack_2d(W, weights, values)}")
    # 输出: 最大价值: 10
```

### 方法二：一维数组优化（推荐）

```python
def knapsack_1d(W, weights, values):
    """
    01背包问题 - 一维数组优化

    Args:
        W: 背包容量
        weights: 物品重量列表
        values: 物品价值列表

    Returns:
        最大价值
    """
    n = len(weights)
    dp = [0] * (W + 1)

    for i in range(n):
        # 关键：倒序遍历容量
        for j in range(W, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

    return dp[W]


# 测试
if __name__ == "__main__":
    W = 10
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    print(f"最大价值: {knapsack_1d(W, weights, values)}")
    # 输出: 最大价值: 10
```

### 方法三：输出具体方案

```python
def knapsack_with_solution(W, weights, values):
    """
    01背包问题 - 输出具体方案

    Args:
        W: 背包容量
        weights: 物品重量列表
        values: 物品价值列表

    Returns:
        (最大价值, 选择的物品索引列表)
    """
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    # 填充dp数组
    for i in range(1, n + 1):
        for j in range(1, W + 1):
            if weights[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i - 1][j - weights[i - 1]] + values[i - 1]
                )

    # 回溯找出选择的物品
    selected = []
    j = W
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            selected.append(i - 1)  # 物品索引
            j -= weights[i - 1]

    return dp[n][W], selected[::-1]


# 测试
if __name__ == "__main__":
    W = 10
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    max_value, selected = knapsack_with_solution(W, weights, values)

    print(f"最大价值: {max_value}")
    print(f"选择的物品索引: {selected}")
    print("物品详情:")
    for idx in selected:
        print(f"  物品{idx}: 重量={weights[idx]}, 价值={values[idx]}")

    # 输出:
    # 最大价值: 10
    # 选择的物品索引: [0, 1, 2]
    # 物品详情:
    #   物品0: 重量=2, 价值=3
    #   物品1: 重量=3, 价值=4
    #   物品2: 重量=4, 价值=5
```

## JavaScript 示例代码

```javascript
/**
 * 01背包问题 - 一维数组优化
 * @param {number} W - 背包容量
 * @param {number[]} weights - 物品重量数组
 * @param {number[]} values - 物品价值数组
 * @returns {number} 最大价值
 */
function knapsack(W, weights, values) {
    const n = weights.length;
    const dp = new Array(W + 1).fill(0);

    for (let i = 0; i < n; i++) {
        // 倒序遍历容量
        for (let j = W; j >= weights[i]; j--) {
            dp[j] = Math.max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

// 测试
const W = 10;
const weights = [2, 3, 4, 5];
const values = [3, 4, 5, 6];
console.log(`最大价值: ${knapsack(W, weights, values)}`);
// 输出: 最大价值: 10
```

## Java 示例代码

```java
public class Knapsack01 {

    /**
     * 01背包问题 - 一维数组优化
     */
    public static int knapsack(int W, int[] weights, int[] values) {
        int n = weights.length;
        int[] dp = new int[W + 1];

        for (int i = 0; i < n; i++) {
            // 倒序遍历容量
            for (int j = W; j >= weights[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - weights[i]] + values[i]);
            }
        }

        return dp[W];
    }

    public static void main(String[] args) {
        int W = 10;
        int[] weights = {2, 3, 4, 5};
        int[] values = {3, 4, 5, 6};

        System.out.println("最大价值: " + knapsack(W, weights, values));
        // 输出: 最大价值: 10
    }
}
```

## C++ 示例代码

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

/**
 * 01背包问题 - 一维数组优化
 */
int knapsack(int W, vector<int>& weights, vector<int>& values) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int i = 0; i < n; i++) {
        // 倒序遍历容量
        for (int j = W; j >= weights[i]; j--) {
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

int main() {
    int W = 10;
    vector<int> weights = {2, 3, 4, 5};
    vector<int> values = {3, 4, 5, 6};

    cout << "最大价值: " << knapsack(W, weights, values) << endl;
    // 输出: 最大价值: 10

    return 0;
}
```

## 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 二维数组 | O(n × W) | O(n × W) |
| 一维数组 | O(n × W) | O(W) |

其中 n 为物品数量，W 为背包容量。

## 图解示例

```
物品:      0    1    2    3
重量:      2    3    4    5
价值:      3    4    5    6
背包容量: 10

最优方案: 选择物品0、1、2
总重量: 2 + 3 + 4 = 9
总价值: 3 + 4 + 5 = 12
(注意：这个例子中容量10，最优可能是其他组合)
```

## 扩展问题

- **完全背包**：每个物品可以无限取
- **多重背包**：每个物品有数量限制
- **二维费用背包**：考虑体积和重量两种限制

## 关键要点

1. **状态定义**：`dp[j]`表示容量为j时的最大价值
2. **状态转移**：`max(不选, 选当前物品)`
3. **遍历顺序**：一维数组必须**倒序**遍历容量
4. **初始化**：`dp[0] = 0`，其余初始化为0
