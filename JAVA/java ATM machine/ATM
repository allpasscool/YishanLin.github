import java.util.Scanner;

public class ATM {
	//variables
	private Account[] accounts;
	private int currentAccountId;
	private Scanner inputScanner;
	
	//constructors
	ATM(Scanner inputScanner)
	{
		this(10, 100.0, inputScanner);
	}
	ATM(int numberOfAccount, double initialBalance, Scanner inputScanner)
	{
		this.accounts = new Account[numberOfAccount];
		for(int i = 0; i < numberOfAccount; i++)
			this.accounts[i] = new Account(i, initialBalance);
		this.inputScanner = inputScanner;
	}
	
	
	//methods
	//getter
	public int getCurrentAccountId()
	{
		return currentAccountId;
	}
	//setter
	public void setCurrentAccountId(int id)
	{
		currentAccountId = id;
	}
	//show menu
	public int showMenu()
	{
		int choice = 0;
		do{
			System.out.println("Main menu");
			System.out.println("1: check balance");
			System.out.println("2: withdraw");
			System.out.println("3: deposit");
			System.out.println("4: exit");
			System.out.print("Enter a choice:");
			choice = inputScanner.nextInt();
			
			if(choice <= 0 || choice >= 5)
				System.out.println("Wrong choice, try again!\n");
		}while(choice <= 0 || choice >= 5);
		return choice;
	}
	//choose account
	public void	chooseAccount()
	{
		System.out.println("Welcome to CS332 ATM!");
		int accoId;
		
		do
		{
			System.out.print("Enter an id: ");
			accoId = inputScanner.nextInt();
			setCurrentAccountId(accoId);
			if(accoId >= accounts.length || accoId < 0)
				System.out.println("Please enter a correct id");
		}while(accoId >= accounts.length || accoId < 0);
		System.out.println("");
	}
	//show balance
	public void displayAccountBalance()
	{
		System.out.printf("Account#%d balance is %.2f\n", getCurrentAccountId(), accounts[currentAccountId].getBalance());
	}
	//withdraw
	public void withdraw()
	{
		double withdraw;
		do
		{
			System.out.print("Enter an amount to withdraw: ");
			withdraw = inputScanner.nextDouble();
			if(withdraw <= accounts[currentAccountId].getBalance() && withdraw > 0)
			{
				accounts[currentAccountId].setBalance(accounts[currentAccountId].getBalance() - withdraw);
				break;
			}
			else
				System.out.println("Invalid amount, ignored");
		}while(true);
	}
	//deposit
	public void deposit()
	{
		double deposit;
		do
		{	
			System.out.print("Enter an amount to deposit: ");
			deposit = inputScanner.nextDouble();
			if(deposit > 0)
			{
				accounts[currentAccountId].setBalance(accounts[currentAccountId].getBalance() + deposit);
				System.out.printf("Account#%d balance is %.2f\n", currentAccountId, accounts[currentAccountId].getBalance());
			}
			else
				System.out.println("Invalid amount, ignored");
		}while(deposit <= 0);
	}
}
