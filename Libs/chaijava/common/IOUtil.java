package chaijava.common;

import java.io.*;
import java.util.*;

/**
 * Collection of methods for doing I/O operations.
 */
public class IOUtil
{
    public static List<String> getLines(String fileName, boolean trim)
            throws IOException {
        List<String> lines = new ArrayList<String>();
        BufferedReader br = new BufferedReader(new FileReader(fileName));

        String line;
        while ((line = br.readLine()) != null)
            lines.add(trim ? line.trim() : line);

        br.close();
        return lines;
    }

    public static List<String> getLines(String fileName) throws IOException {
        return getLines(fileName, true);
    }

    /**
     * Read a dictionary (CMU format) into a mapping from word to a list of
     * pronunciations, where keys are sorted in alphabetical order.
     */
    public static Map<String, List<String>> loadDict(String filePath)
            throws IOException {
        Map<String, List<String>> dict = new TreeMap<String, List<String>>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));

        String line;
        while ((line = br.readLine()) != null) {
            String[] ary = line.split("\t");
            String word = ary[0].trim();
            String pronunciation = ary[1].trim();

            if (!dict.containsKey(word))
                dict.put(word, new ArrayList<String>());
            dict.get(word).add(pronunciation);
        }

        br.close();
        return dict;
    }
}
