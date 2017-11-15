import java.util.*;

public class Ex11_1 {
	//print out menu
	public static int getMenuOption(Scanner inputScanner) throws IllegalArgumentException
	{
		if(inputScanner != null)
		{
			while(true)
			{
				try
				{
					//get the option
					System.out.print("(1)Add (2)Sort (3)Print (0)Exit: ");
					int option = inputScanner.nextInt();
					if(option < 0 || option >= 4)
						System.out.println("Incorrect input: 0, 1, 2, or 3 is required\n");
					else
						return option;
				}
				//when input error occurs
				catch(InputMismatchException ex)
				{
					System.out.println("Incorrect input: 0, 1, 2, or 3 is required\n");
					inputScanner.nextLine();
				}
			}
		}
		//when argument have problem
		else
		{
			throw new IllegalArgumentException("Incorrect input: 0, 1, 2, or 3 is required");
		}
	}
	
	//add triangle to the list
	public static void addTriangle(Scanner inputScanner,
			ArrayList<Triangle> list) throws IllegalArgumentException
	{
		if(inputScanner != null || list != null)
		{ 
			Triangle tri;
			while(true)
			{
				//add triangle
				try
				{
					System.out.print("Enter three sides: ");
					tri = new Triangle(inputScanner.nextDouble(), inputScanner.nextDouble(), inputScanner.nextDouble());
					break;
				}
				//when input error happens
				catch(InputMismatchException ex)
				{
					System.out.println("Incorrect input: double value is required");
					inputScanner.nextLine();
				}
				//when arguments have problem
				catch(IllegalArgumentException ex1)
				{
					System.out.println(ex1.getMessage());
				}
			}
			//decide color
			System.out.print("Enter the color: ");
			inputScanner.nextLine();
			tri.setColor(inputScanner.nextLine());
			while(true)
			{
				//decide if it is filled
				try
				{
					System.out.print("Enter a boolean value for filled: ");
					tri.setFilled(inputScanner.nextBoolean());
					break;
				}
				//when input error happens
				catch(InputMismatchException ex1)
				{
					System.out.println("Incorrect input: boolean value is required ");
					inputScanner.nextLine();
				}
			}
			list.add(tri);
			System.out.println("");
		}
		//when arguments have problem
		else 
			throw new IllegalArgumentException("");
	
	}
	
	//insertion sort the list by their area
	public static void sortTriangles(ArrayList<Triangle> list)
	{
		Triangle triTem;
		for(int i = 1; i < list.size(); i ++)
		{
			for(int j = 0; j < list.size(); j++)
			{
				if(list.get(i).getArea() < list.get(j).getArea())
				{
					triTem = list.get(i);
					for(int k = i; k > j; k--)
					{
						list.set(k, list.get(k - 1));
					}
					list.set(j, triTem);
				}
			}
		}
		
		System.out.println("Insertion sort performed!\n");
	}
	
	//print out the triangle(s) in the list
	public static void printTriangles(ArrayList<Triangle> list)
	{
		switch(list.size())
		{
		case 0:
			System.out.println("The triangle list is empty!\n");
			break;
		case 1:
			System.out.println("The triangle list has 1 triangle:\n" + list.get(0).toString() + "\n");
			break;
		default:
			System.out.println("The triangle list has " + list.size() + " triangles: ");
			for(int i = 0; i < list.size(); i++)
			{
				System.out.println(list.get(i).toString());
			}
			System.out.println("");
		}
	}
	
	public static void main(String[] args)
	{
		ArrayList<Triangle> list = new ArrayList<Triangle>();
		System.out.println("CS332 Triangle List\n");
		Scanner input = new Scanner(System.in);
		int ex;
		// 1 is add, 2 is sort, 3 is print, 0 is exit
		do
		{
			ex = getMenuOption(input);
			switch(ex)
			{
			case 1:
				addTriangle(input, list);
				break;
			case 2:
				sortTriangles(list);
				break;
			case 3:
				printTriangles(list);
			}
		} while(ex != 0);
		System.out.println("Goodbye!");
		input.close();

	}
}
