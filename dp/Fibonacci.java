package dp;


public class Fibonacci {


    private static long frb(int n) {

        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        long[] dp = new long[n + 1];
        dp[0] = 0;
        dp[1] = 1;

        for (int i = 2; i < n + 1; i++) {
            dp[i] = dp[i - 2] + dp[i - 1];
        }

        return dp[n];
    }

    public static void main(String[] args) {
        System.out.println(frb(0));
        System.out.println(frb(1));
        System.out.println(frb(2));
        System.out.println(frb(3));
        System.out.println(frb(4));
        System.out.println(frb(5));
        System.out.println(frb(6));
        System.out.println(frb(50));

    }


}