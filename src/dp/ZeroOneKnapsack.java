package src.dp;

/**
 * 01背包问题
 *
 * 题目描述：
 * 有 n 件物品和一个容量为 W 的背包。
 * 第 i 件物品的重量是 weights[i]，价值是 values[i]。
 * 每件物品只能选择放或不放（0或1），求在不超过背包容量的情况下，能获得的最大价值。
 *
 * 示例：
 * 物品重量: [2, 3, 4, 5]
 * 物品价值: [3, 4, 5, 6]
 * 背包容量: 5
 * 结果: 7 (选择重量2和3的物品，价值3+4=7)
 */

public class ZeroOneKnapsack {

    /**
     * 二维动态规划解法
     * dp[i][j] 表示前 i 件物品放入容量为 j 的背包能获得的最大价值
     *
     * 时间复杂度: O(n * W)
     * 空间复杂度: O(n * W)
     */
    public static int knapsack2D(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        int[][] dp = new int[n + 1][capacity + 1];

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= capacity; j++) {
                // 不放第 i 件物品
                dp[i][j] = dp[i - 1][j];
                // 如果放得下第 i 件物品，选择放与不放中的最大值
                if (j >= weights[i - 1]) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - weights[i - 1]] + values[i - 1]);
                }
            }
        }

        return dp[n][capacity];
    }

    /**
     * 一维动态规划解法（空间优化）
     * dp[j] 表示容量为 j 的背包能获得的最大价值
     *
     * 时间复杂度: O(n * W)
     * 空间复杂度: O(W)
     */
    public static int knapsack1D(int[] weights, int[] values, int capacity) {
        int[] dp = new int[capacity + 1];

        for (int i = 0; i < weights.length; i++) {
            // 从后向前遍历，避免重复使用同一件物品
            for (int j = capacity; j >= weights[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - weights[i]] + values[i]);
            }
        }

        return dp[capacity];
    }

    /**
     * 带回溯的解法，返回具体选择的物品
     */
    public static int knapsackWithPath(int[] weights, int[] values, int capacity, boolean[] selected) {
        int n = weights.length;
        int[][] dp = new int[n + 1][capacity + 1];

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= capacity; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= weights[i - 1]) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - weights[i - 1]] + values[i - 1]);
                }
            }
        }

        // 回溯找出选择的物品
        int j = capacity;
        for (int i = n; i >= 1; i--) {
            if (dp[i][j] != dp[i - 1][j]) {
                selected[i - 1] = true;
                j -= weights[i - 1];
            }
        }

        return dp[n][capacity];
    }

    static void main() {
        // 测试用例
        int[] weights = {2, 3, 4, 5};
        int[] values = {3, 4, 5, 6};
        int capacity = 5;

        System.out.println("01背包问题测试");
        System.out.println("物品重量: " + java.util.Arrays.toString(weights));
        System.out.println("物品价值: " + java.util.Arrays.toString(values));
        System.out.println("背包容量: " + capacity);
        System.out.println();

        // 二维DP解法
        int result2D = knapsack2D(weights, values, capacity);
        System.out.println("二维DP解法 - 最大价值: " + result2D);

        // 一维DP解法
        int result1D = knapsack1D(weights, values, capacity);
        System.out.println("一维DP解法 - 最大价值: " + result1D);

        // 带回溯解法
        boolean[] selected = new boolean[weights.length];
        int resultWithPath = knapsackWithPath(weights, values, capacity, selected);
        System.out.println("带回溯解法 - 最大价值: " + resultWithPath);

        System.out.print("选择的物品: ");
        int totalWeight = 0;
        for (int i = 0; i < selected.length; i++) {
            if (selected[i]) {
                System.out.print("物品" + (i + 1) + "(重量" + weights[i] + ",价值" + values[i] + ") ");
                totalWeight += weights[i];
            }
        }
        System.out.println();
        System.out.println("总重量: " + totalWeight + "/" + capacity);
    }
}
