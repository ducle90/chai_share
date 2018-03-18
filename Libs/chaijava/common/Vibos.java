package chaijava.common;

import java.util.*;

/**
 * Very Important but Obscure Stuff (Vibos), or just a fancy way to say Utils.
 */
public class Vibos
{
    public static <T extends Object> Set<T> intersection(Set<T> x, Set<T> y) {
        if (x == null && y == null)
            return null;
        else if (x == null)
            return y;
        else if (y == null)
            return x;
        else {
            Set<T> intersection = new HashSet<T>();
            for (T t : x) {
                if (y.contains(t))
                    intersection.add(t);
            }
            return intersection;
        }
    }

    public static <T extends Object> Set<T> merge(Set<T> set1, Set<T> set2) {
        Set<T> merged = new HashSet<T>(set1);
        for (T t : set2)
            merged.add(t);
        return merged;
    }

    public static <X extends Object, Y extends Object> Object[] concat(X[] x, Y[] y) {
        Object[] concat = new Object[x.length + y.length];
        // Copy x
        for (int i = 0; i < x.length; i++)
            concat[i] = x[i];
        // Copy y
        for (int i = 0; i < y.length; i++)
            concat[i + x.length] = y[i];
        return concat;
    }

    public static <T extends Object> T firstOf(List<T> list) {
        return list.get(0);
    }

    public static <T extends Object> T lastOf(List<T> list) {
        return list.get(list.size() - 1);
    }

    public static int asInt(boolean b) {
        return b ? 1 : 0;
    }

    public static String removeExtension(String fileName) {
        int lastIndex = fileName.lastIndexOf(".");
        return lastIndex == -1 ? fileName : fileName.substring(0, lastIndex);
    }

    public static String removePath(String fileName) {
        String[] ary = fileName.split("/");
        return ary[ary.length - 1];
    }

    public static <T extends Object> int indexOf(T toFind, T[] ary) {
        for (int i = 0; i < ary.length; i++) {
            if (toFind.equals(ary[i]))
                return i;
        }
        return -1;
    }

    public static <T extends Object> int lastIndexOf(T toFind, T[] ary) {
        for (int i = ary.length - 1; i >= 0; i--) {
            if (toFind.equals(ary[i]))
                return i;
        }
        return -1;
    }

    public static <T extends Object> boolean contains(T toFind, T[] ary) {
        return indexOf(toFind, ary) != -1;
    }
}
