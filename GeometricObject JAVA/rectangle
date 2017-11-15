public class Rectangle extends GeometricObject{
	//date field
	private double width;
	private double height;
	
	//constructors
	public Rectangle(double width, double height)
	{
		this.width = width;
		this.height = height;
	}
	public Rectangle(double width, double height, String color, boolean filled)
	{
		this.width = width;
		this.height = height;
		setColor(color);
		setFilled(filled);
	}
	
	//getter
	public double getWidth()
	{
		return width;
	}
	public void setWidth(double width)
	{
		this.width = width;
	}
	public double getHeight()
	{
		return height;
	}
	//setter
	private void setHeight(double height)
	{
		this.height = height;
	}
	public double getArea()
	{
		return width * height;
	}
	//get perimeter
	public double getPerimeter(){
		return 2 * (width + height);
	}
}