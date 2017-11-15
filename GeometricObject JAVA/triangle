
public class Triangle extends GeometricObject{
	//date fields
	private double side1;
	private double side2;
	private double side3;
	
	//constructors
	public Triangle()
	{
		this(1.0, 1.0, 1.0);
	}
	public Triangle(double side1, double side2, double side3) throws IllegalArgumentException
	{
		if(side1 > 0 && side2 > 0 && side3 > 0 && ((side1 + side2 > side3) || (side2 + side3 > side1) || (side1 + side3 > side2)))
		{
			this.side1 = side1;
			this.side2 = side2;
			this.side3 = side3;
		}
		else
		{
			throw new IllegalArgumentException(
					"Invalid sides for triangle: (" + side1 + ", " + side2 + ", " + side3 + ")");
				
		}
	}
	
	//setters
	public void setSide1(double side1)
	{
		this.side1 = side1;
	}
	public void setSide2(double side2)
	{
		this.side2 = side2;
	}
	public void setSide3(double side3)
	{
		this.side3 = side3;
	}
	//getters
	public double getArea()
	{
		double p = (side1 + side2 + side3) / 2;
		return Math.sqrt(p * (p - side1) * (p - side2) * (p-side3));
	}
	//get perimeter
	public double getPerimeter()
	{
		return side1 + side2 + side3;
	}

	//override toString
	public String toString()
	{
		String fill = "not filled";
		if(isFilled())
			fill = "filled";
		String s = "Triangle (" + side1 + ", " + side2 + ",  " + side3 + "): "
				+ getColor() + ", " + fill + ", " + "area = " + getArea();
		return s;
	}
}