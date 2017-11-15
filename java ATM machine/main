import java.util.Scanner;

public class Ex10_7 {
	public static void main(String[] args)
	{
		Scanner input = new Scanner(System.in);
		ATM atm = new ATM(input);
		//infinite loop
		while(true)
		{
			//choose account
			atm.chooseAccount();
			boolean exit = false;
			do
			{
				//decide the behavior
				switch (atm.showMenu()){
				case 1:
					atm.displayAccountBalance();
					break;
				case 2:
					atm.withdraw();
					break;
				case 3:
					atm.deposit();
					break;
				case 4:
					exit = true;
					break;
				}
				System.out.println("");
			
			}while(!exit);
		}
	}
}
