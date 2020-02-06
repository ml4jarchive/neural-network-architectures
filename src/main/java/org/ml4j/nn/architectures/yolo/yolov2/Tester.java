package org.ml4j.nn.architectures.yolo.yolov2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Tester {

	public static void main(String[] args) throws FileNotFoundException {
		
		Scanner scanner = new Scanner(new File("/Users/michael/Desktop/convout2"));
		   scanner.useDelimiter(" ");
		   int count = 0;
	        while(scanner.hasNext()){
	            //System.out.print(scanner.next()+"|");
	        	scanner.next();
	        	count++;
	        }
	        scanner.close();
	        System.out.println(count);
	}
}
