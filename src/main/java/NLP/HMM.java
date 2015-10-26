/**
 * @author kvivekanandan
 * Oct 24, 2015
 * HMM.java
 */

package NLP;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;

public class HMM {

	public static List<List<Double>> stateMatrix = new ArrayList<List<Double>>();
	public static List<List<Double>> emissonMatrix = new ArrayList<List<Double>>();
	public static HashMap<String, Double> initStateProb = new HashMap<String, Double>();
	public static HashMap<String, Integer> states = new HashMap<String, Integer>();
	public static HashMap<String, Integer> observations = new HashMap<String, Integer>();
	public static List<List<Double>> viterbiMatrix = new ArrayList<List<Double>>();


	public static void calculateHMMParameters(String path) {
		getTrainStatesAndObservations(path);
		calculateMatrices(path);
	}
	
	public static void getTrainStatesAndObservations(String path) {
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
			String line = null;
			int current_obs_num = 0;
			int current_state_num = 0;
			states.put("###", current_state_num++);
			observations.put("###", current_obs_num++);
			while ((line = reader.readLine()) != null) {
				if (line.equals("###/###")) {
					continue;
				}
				String[] l = line.split("/");
				if (l != null && l.length == 2) {
					String new_obs = l[0];
					String new_st = l[1];
					if (!observations.containsKey(new_obs)) {
						observations.put(new_obs, current_obs_num++);
					}
					if (!states.containsKey(new_st)) {
						states.put(new_st, current_state_num++);
					}
				}
			}

			observations.put("UNK", current_obs_num++);
			for (int i = 0; i < states.size(); i++) {
				List<Double> list = new ArrayList<Double>(Collections.nCopies(states.size(), 1.0));
				stateMatrix.add(list);
			}
			for (int i = 0; i < states.size(); i++) {
				List<Double> list = new ArrayList<Double>(Collections.nCopies(observations.size(), 1.0));
				emissonMatrix.add(list);
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null)
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		}
	}


	public static void calculateMatrices(String path) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
			String line = null;
			Integer current_state = null;
			Integer current_obs = null;
			Integer previous_state = states.get("###");
			HashMap<String, Integer> unseenObs = new HashMap<String, Integer>(observations);
			while ((line = reader.readLine()) != null) {
				String[] l = line.split("/");
				if (l != null && l.length == 2) {
					String obs = l[0];
					String cs = l[1];
					current_state = states.get(cs);
					Double v = stateMatrix.get(previous_state).get(current_state);
					v += 1.0;
					stateMatrix.get(previous_state).set(current_state, v);

					if (unseenObs.containsKey(obs)) {
						unseenObs.remove(obs);
						obs = "UNK";
					}
					current_obs = observations.get(obs);
					Double o = emissonMatrix.get(current_state).get(current_obs);
					o += 1.0;
					emissonMatrix.get(current_state).set(current_obs, o);
				}
				previous_state = current_state;
			}

			//
			// for(int i=0;i<stateMatrix.size();i++){
			// for(int j=0;j<stateMatrix.get(i).size();j++){
			// double p = stateMatrix.get(i).get(j) ;
			// stateMatrix.get(i).set(j,p+1);
			// }
			// }

			HashMap<Integer, Double> marginalEmissonProb = new HashMap<Integer, Double>();
			for (int i = 0; i < emissonMatrix.size(); i++) {
				double sum = 0;
				for (int j = 0; j < emissonMatrix.get(i).size(); j++) {
					sum += emissonMatrix.get(i).get(j);
				}
				marginalEmissonProb.put(i, sum);
			}
			for (int i = 0; i < emissonMatrix.size(); i++) {
				double sum = 0;
				for (int j = 0; j < emissonMatrix.get(i).size(); j++) {
					double p = emissonMatrix.get(i).get(j) / marginalEmissonProb.get(i);
					emissonMatrix.get(i).set(j, p);
				}
			}

			HashMap<Integer, Double> marginalProb = new HashMap<Integer, Double>();
			for (int i = 0; i < stateMatrix.size(); i++) {
				double sum = 0;
				for (int j = 0; j < stateMatrix.get(i).size(); j++) {
					sum += stateMatrix.get(i).get(j);
				}
				marginalProb.put(i, sum);
			}
			for (int i = 0; i < stateMatrix.size(); i++) {
				double sum = 0;
				for (int j = 0; j < stateMatrix.get(i).size(); j++) {
					double p = stateMatrix.get(i).get(j) / marginalProb.get(i);
					stateMatrix.get(i).set(j, p);
				}
			}

			for (int i = 0; i < stateMatrix.get(0).size(); i++) {
				initStateProb.put(getKeyByValue(states, i), stateMatrix.get(0).get(i));
			}
			// printMap(initStateProb);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static List<String> predict(ArrayList<String> obsSeq) {
		return Viterbi.maxLikelyStateSequence(obsSeq, stateMatrix, emissonMatrix, states, observations, initStateProb);
	}

	public static <T> void printMatrix(List<List<T>> matrix, Map<String, Integer> map) {
		for (int i = 0; i < matrix.size(); i++) {
			if (i == 0)
				System.out.print(getKeyByValue(map, i) + " ");
			else
				System.out.print(getKeyByValue(map, i) + "   ");
			for (int j = 0; j < matrix.get(i).size(); j++) {
				if (matrix.get(i).get(j) instanceof Double)
					System.out.print(String.format("%.20f", matrix.get(i).get(j)));
				else
					System.out.print(matrix.get(i).get(j) + " ");
			}
			System.out.println();
		}
	}

	public static <T, E> T getKeyByValue(Map<T, E> map, E value) {
		for (Entry<T, E> entry : map.entrySet()) {
			if (Objects.equals(value, entry.getValue())) {
				return entry.getKey();
			}
		}
		return null;
	}

	public static <K, V> void printMap(HashMap<K, V> map) {
		for (Entry e : map.entrySet()) {
			System.out.println(e.getKey() + " " + e.getValue());
		}

	}

	public static void predictTestData(String path) {
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
			String line = null;
			ArrayList<String> input = new ArrayList<String>();
			ArrayList<String> input_tags = new ArrayList<String>();
			List<String> output = new ArrayList<String>();
			Double num_words_test_set = 0.0;
			Double num_correctclassified_words = 0.0;
			boolean isEnd = false;
			while ((line = reader.readLine()) != null) {
				String[] l = line.split("/");

				if (l != null && l.length == 2) {

					if (line.contains("###")) {
						if (isEnd) {
							output = predict(input);
							if (input.size() != output.size()) {
								System.out.println("INPUT SIZE NOT EQUAL TO OUTPUT SIZE");
								System.exit(0);
							}
							for (int i = 0; i < input.size(); i++) {
								if (input_tags.get(i).equals(output.get(i))) {
									num_correctclassified_words += 1.0;
								}
							}
							isEnd = false;
							input = new ArrayList<String>();
							input_tags = new ArrayList<String>();
						} else {
							isEnd = true;
						}
					} else {
						input_tags.add(l[0] + "/" + l[1]);
						input.add(l[0]);
						num_words_test_set += 1.0;
					}
				}
			}

			double error_rate = num_correctclassified_words / num_words_test_set;
			System.out.println("ERROR RATE : " + error_rate);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}

	public static void main(String args[]) {
		calculateHMMParameters("entrain.txt");
		predictTestData("entest.txt");
	}

}
