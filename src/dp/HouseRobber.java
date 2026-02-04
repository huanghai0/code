package src.dp;

public class HouseRobber {

    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }

        int prev1 = 0; // Max money that can be robbed up to the previous house
        int prev2 = 0; // Max money that can be robbed up to the house before the previous house

        for (int num : nums) {
            int temp = prev1;
            prev1 = Math.max(prev2 + num, prev1);
            prev2 = temp;
        }

        return prev1;
    }

    private int houseRobber(int[] nums) {

        if (nums == null || nums.length < 1) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);

        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[dp.length - 1];
    }

    public static void main(String[] args) {
        HouseRobber solution = new HouseRobber();
        int[] nums = { 2, 7, 9, 3, 1 };
        int result = solution.houseRobber(nums);
        System.out.println(result); // Output: 12
    }
}
