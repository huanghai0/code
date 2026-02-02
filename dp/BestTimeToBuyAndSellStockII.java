package dp;

public class BestTimeToBuyAndSellStockII {

    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }

        int totalProfit = 0;

        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                totalProfit += prices[i] - prices[i - 1];
            }
        }

        return totalProfit;
    }

    private int sell(int[] prices) {

        if (prices == null || prices.length < 1) {
            return 0;
        }

        int n = prices.length;
        int[] holds = new int[n];
        int[] sells = new int[n];

        holds[0] = -prices[0];
        sells[0] = 0;

        for (int i = 1; i < n; i++) {

            holds[i] = Math.max(holds[i - 1], sells[i - 1] - prices[i]);

            sells[i] = Math.max(sells[i - 1], holds[i - 1] + prices[i]);
        }

        return sells[n - 1];
    }

    public static void main(String[] args) {
        BestTimeToBuyAndSellStockII solution = new BestTimeToBuyAndSellStockII();
        int[] prices = { 7, 1, 5, 3, 6, 4 };
        int result = solution.sell(prices);
        System.out.println(result); // Output: 7
    }
}
