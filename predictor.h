#ifndef PREDICTOR_H
#define PREDICTOR_H
#include "reader.h"
#include "data.h"
#include "output.h"
#include "stat.h"
#include <map>
#include "random_generator.h"
#include <algorithm>
#include <string>
#include <math.h>
#include <cstdlib>
#include <stopwatch.h>

class predictor
{
public:
    predictor(string xprediction_file_path, data * xdt);

private:
    struct predicted_values
    {
        predicted_values(double xoccured, double xpredicted_approach, double xpredicted_allst, double xpredicted_hmm, double xpercentage_error_approach,
                         double xpercentage_error_aalst, double xpercentage_error_hmm):
                        occured(xoccured), predicted_approach(xpredicted_approach), predicted_allst(xpredicted_allst), predicted_hmm(xpredicted_hmm),
                        percentage_error_approach(xpercentage_error_approach), percentage_error_aalst(xpercentage_error_aalst), percentage_error_hmm(xpercentage_error_hmm)
        {
        }
        double occured = 0;
        double predicted_approach = 0;
        double predicted_allst = 0;
        double predicted_hmm = 0;

        double percentage_error_approach = 0;
        double percentage_error_aalst = 0;
        double percentage_error_hmm = 0;

        double difference_obs_predicted_approach = 0;
        double difference_obs_predicted_allst = 0;
        double difference_obs_predicted_hmm = 0;

    };

    struct estimate_errors
    {
        double MAPE_hmm = 0;
        double MAPE_aalst = 0;
        double MAPE_approach = 0; 
        double st_deviation_hmm = 0;
        double st_deviation_allst = 0;
        double st_deviation_approach = 0;
        double MAE_hmm = 0;
        double MAE_aalst = 0;
        double MAE_approach = 0;  
        double RMSE_hmm = 0;
        double RMSE_aalst = 0;
        double RMSE_approach = 0;
    };

    data * d;
    reader r;
    output o;
    stats st;
    random_generator rg;
    vector<data::trace> v_randomized_traces;
    int log_size;
    string prediciton_file_path;

    map<string, int> m_states;
    map<int, string> m_int_states;
    vector<vector<double>> A;
    vector<vector<double>> B;
    vector<double> pi;
    vector<vector<int>> xO;
    vector<vector<int>> state_from_to_matrix;
    void calculate_trace_frequency();
    void print_trace_frequency();
    void create_complete_transition_system(vector<data::trace> &xlog);
    void create_product_transition_system(vector<data::trace> &xlog);
    void separate_k_fold_log(vector<data::trace> &xcomplete_log, vector<data::trace> &xtraining_log, vector<data::trace> &xtest_log, int xtotal_fold, int xcurrent_fold);
    void creates_transition_system();
    void creates_transition_systems();
    map<string,data::state_data> creates_transition_system(map<string,data::trace> &t, double model_rate);
    void calculate_state_mean(map<string, data::state_data> &t);
    int calculate_remaining_time();
    void calculate_state_mean();
    void test_prediction();
    void make_prediciton_k_fold(int xfold_numbers);
    void predict_values(vector<data::trace> &xtest_log, map<string, map<string, vector<predicted_values> > > &xmm_predicted_data);
    void create_randomized_log();
    string create_prediction_state_string(double &real_time, data::trace &t, int string_size);
    void generates_artificial_log();
    void calculate_errors(map<string, map<string, vector<predicted_values> > > &xmm_predicted_data, map<string, map<string, estimate_errors>> &xm_errors);
    void print_errors(string xfile_path, map<string, map<string, estimate_errors>> &xm_errors);
    void initialize_markov();
    void create_states_map();
    void create_observation_matrix();
    void initialize_markov_parameters();
    double alpha(int xt, int xi, vector<int> O);
    double P_observation(vector<int> O);
    double betha(int xt, int xi, vector<int> O);
    double epsilon(int xt, int xi, int xj, vector<int> O);
    double gamma(int xt, int xi, vector<int> O);
    double gamma_sum(int xi, vector<int> O);
    double epsilon_sum(int xi, int xj, vector<int> O);
    void estimate_markov_model_parameters(int training_data_size, int convergence_limit);
    vector<double> estimate_pi(int training_data_size);
    vector<vector<double>> estimate_A(int training_data_size);
    vector<vector<double>> estimate_B(int training_data_size);
    void print_parameters(int i, double ellapsed_time);
    void read_markov_parameters(string parameters_file);
    void create_state_from_to_matrix();
    double predict_markov_value(data::trace &t, int string_size, string prediction_string);
};

#endif // PREDICTOR_H
predictor::predictor(string xprediction_file_path, data * xd): prediciton_file_path(xprediction_file_path), d(xd)
{
    create_randomized_log();
    make_prediciton_k_fold(10);


}

void predictor::create_complete_transition_system(vector<data::trace> &xlog)
{
    d->complete_transition_system = map<string, data::state_data>();
    data::state_data state;
    d->complete_transition_system[""] = state;
    for(vector<data::trace>::iterator it = xlog.begin(); it != xlog.end(); it++)
    {
        int total_time_by_product = static_cast<int>(it->total_trace_time/it->events.back().product_qtty);
        d->complete_transition_system[""].time_to_finish.push_back(total_time_by_product);
    }
    for(vector<data::trace>::iterator it = xlog.begin(); it != xlog.end(); it++)
    {
        string state = "";
        for(list<data::event>::iterator it_event = it->events.begin(); it_event != it->events.end(); it_event++)
        {
            state += it_event->activity;
            int completion_time = static_cast<int>((it->events.back().d2 - it_event->d2)/it_event->product_qtty);
            if(d->complete_transition_system.find(state) == d->complete_transition_system.end()) //the state does not exist yet
            {
                data::state_data a;
                a.time_to_finish.push_back(completion_time);
                d->complete_transition_system[state] = a;
            }else
            {
                d->complete_transition_system[state].time_to_finish.push_back(completion_time);
            }
        }
    }
    calculate_state_mean(d->complete_transition_system);
}

void predictor::create_product_transition_system(vector<data::trace> &xlog)
{
    d->transition_systems = map<string, map<string,data::state_data>>();
    for(vector<data::trace>::iterator it = xlog.begin(); it != xlog.end(); it++)
    {
        int total_time_by_product = static_cast<int>((it->total_trace_time)/it->events.back().product_qtty);
        string product = std::to_string(it->events.back().product);
        d->transition_systems[product][""].time_to_finish.push_back(total_time_by_product);
    }
    for(vector<data::trace>::iterator it = xlog.begin(); it != xlog.end(); it++)
    {
        string state = "";
        string product = std::to_string(it->events.back().product);
        for(list<data::event>::iterator it_event = it->events.begin(); it_event != it->events.end(); it_event++)
        {
            state += it_event->activity;
            int completion_time = static_cast<int>((it->events.back().d2 - it_event->d2)/it->events.back().product_qtty);
            if(d->transition_systems[product].find(state) == d->transition_systems[product].end()) //the state does not exist yet
            {
                data::state_data a;
                a.time_to_finish.push_back(completion_time);
                d->transition_systems[product][state] = a;
            }else
            {
                d->transition_systems[product][state].time_to_finish.push_back(completion_time);
            }
        }
    }
    for(map<string, map<string,data::state_data>>::iterator it = d->transition_systems.begin(); it != d->transition_systems.end(); it++)
    {
         calculate_state_mean(it->second);
    }
}

void predictor::separate_k_fold_log(vector<data::trace> &xcomplete_log, vector<data::trace> &xtraining_log, vector<data::trace> &xtest_log, int xtotal_fold, int xcurrent_fold)
{
    double test_range = floor(static_cast<double>(xcomplete_log.size())/static_cast<double>(xtotal_fold));

    int test_initial_value = static_cast<int>(xcurrent_fold * test_range);
    int test_final_value = static_cast<int>(test_initial_value + test_range);
    xtraining_log = vector<data::trace>();
    xtest_log = vector<data::trace>();
    for(int i = 0; i < test_initial_value; i++)
    {
        xtest_log.push_back(xcomplete_log[i]);
    }
    for(int i = test_final_value; i < xcomplete_log.size(); i++)
    {
        xtest_log.push_back(xcomplete_log[i]);
    }
    for(int i = test_initial_value; i < test_final_value; i++)
    {
        xtraining_log.push_back(xcomplete_log[i]);
    }
}
void predictor::creates_transition_systems()
{
    map<string,map<string,data::trace>>::iterator it = d->product_traces.begin();
    while(it != d->product_traces.end())
    {
        d->transition_systems[it->first] = creates_transition_system(it->second,0.97);
        it++;
    }
    calculate_state_mean(d->complete_transition_system);
}
map<string, data::state_data> predictor::creates_transition_system(map<string, data::trace> &t, double model_rate)
{

    int model_trace_limit = t.size()*model_rate;
    int model_count = 0;

    map<string,data::state_data> transition_system;
    map<string,data::trace>::iterator it;
    it = t.begin();
    data::state_data state;
    transition_system[""] = state;
    while(it != t.end())     {
        if(model_count < model_trace_limit)
        {
            d->complete_transition_system[""].time_to_finish.push_back(static_cast<int>(it->second.total_trace_time));
            transition_system[""].time_to_finish.push_back(static_cast<int>(it->second.total_trace_time));
        }else
        {
            d->test_traces[it->first] = it->second;
        }
        model_count ++;
        it++;
    }
    model_count = 0;
    it = t.begin();
    while(it != t.end())
    {
        if(model_count<model_trace_limit)
        {
            list<data::event>::iterator it_2 = it->second.events.begin();
            string state = "";
            while(it_2 != it->second.events.end())
            {
                state += it_2->activity;
                int completion_time = static_cast<int>(it->second.events.back().d2-it_2->d2);
                if(transition_system.find(state) == transition_system.end()) 
                {
                    data::state_data a;
                    a.time_to_finish.push_back(completion_time);
                    transition_system[state] = a;
                     if(d->complete_transition_system.find(state) == transition_system.end())
                     {
                         d->complete_transition_system[state] = a;
                     }else
                     {
                         d->complete_transition_system[state].time_to_finish.push_back(completion_time);
                     }
                }else
                {
                    d->complete_transition_system[state].time_to_finish.push_back(completion_time);
                    transition_system[state].time_to_finish.push_back(completion_time);
                }
                it_2++;
             }
        }else
        {
                d->test_traces[it->first] = it->second;
        }
        model_count++;
        it++;
    }
    calculate_state_mean(transition_system);

    return transition_system;
}

void predictor::calculate_state_mean(map<string,data::state_data> &t)
{
    map<string,data::state_data>::iterator it = t.begin();
    while(it != t.end())
    {
        it->second.mean = st.mean(it->second.time_to_finish);
        it++;
    }
}

void predictor::test_prediction()
{
    ofstream o(prediciton_file_path.c_str());
    o << "Produto;Estado;Estimado_Separado_Produtos;Estimado_Total;Ocorrido \n";
    map<string,data::trace>::iterator it = d->test_traces.begin();
    while(it != d->test_traces.end())
    {
        string product = std::to_string(it->second.events.front().product);
        string estimate_time_to_completion;
        string estimate_to_completion_total;
        double real_time_to_completion = 0;
        string state ;
        estimate_time_to_completion = std::to_string(d->transition_systems[product][state].mean);
        estimate_to_completion_total = std::to_string(d->complete_transition_system[state].mean);
        replace(estimate_time_to_completion.begin(),estimate_time_to_completion.end(),'.',',');
        replace(estimate_to_completion_total.begin(),estimate_to_completion_total.end(),'.',',');
        o << product << ";" << state << ";" <<estimate_time_to_completion << ";" <<estimate_to_completion_total << ";" << real_time_to_completion << "\n";
        it++;
    }

}

void predictor::make_prediciton_k_fold(int xfold_numbers)
{
    int fold = xfold_numbers;
    map<string, map<string, vector<predicted_values>>> mm_predicted_data;
    map<string, map<string, estimate_errors>> mm_estimates_errors;
    vector<data::trace> v_training_log;
    vector<data::trace> v_test_log;
    for(int i = 0; i < fold; i++)
    {
        cout << "Current fold:" << i << "\n";
        separate_k_fold_log(v_randomized_traces, v_training_log, v_test_log, fold, i);
        create_complete_transition_system(v_training_log);
        create_product_transition_system(v_training_log);
        predict_values(v_test_log, mm_predicted_data);
    }
    calculate_errors(mm_predicted_data,  mm_estimates_errors);
    print_errors(prediciton_file_path, mm_estimates_errors);
}

void predictor::predict_values(vector<data::trace> &xtest_log, map<string, map<string, vector<predicted_values>>> &xmm_predicted_data )
{
    for(int i = 0; i < xtest_log.size(); i++)
    {
        string product = std::to_string(xtest_log[i].events.front().product);
        double estimate_time_to_completion_product_transition = 0;
        double estimate_to_completion_complete_transition = 0;
        double estimate_to_completion_hmm = 0;
        double real_time_to_completion = 0;

        int string_size = rg.random_int(xtest_log[i].events.size() - 1);
        string state = create_prediction_state_string(real_time_to_completion, xtest_log[i], string_size);

        estimate_time_to_completion_product_transition = d->transition_systems[product][state].mean; //PROBLEMA AQUI! RECUPERAR O VALOR DA MEDIA ESTÁ AUMENTANDO OS ELEMENTOS DO TRANSITION SYSTEM
        estimate_to_completion_complete_transition = d->complete_transition_system[state].mean;
        estimate_to_completion_hmm = predict_markov_value(xtest_log[i], string_size, state);

        double percentage_error_approach = abs((real_time_to_completion - estimate_time_to_completion_product_transition)/real_time_to_completion);
        double percentage_error_aalst    = abs((real_time_to_completion - estimate_to_completion_complete_transition)/real_time_to_completion);
        double percentage_error_hmm      = abs((real_time_to_completion - estimate_to_completion_hmm)/real_time_to_completion);

        double difference_obs_predicted_approach = abs((real_time_to_completion - estimate_time_to_completion_product_transition));
        double difference_obs_predicted_allst    = abs((real_time_to_completion - estimate_to_completion_complete_transition));
        double difference_obs_predicted_hmm      = abs((real_time_to_completion - estimate_to_completion_hmm));

        predicted_values p(real_time_to_completion, estimate_time_to_completion_product_transition, estimate_to_completion_complete_transition, estimate_to_completion_hmm,
                           percentage_error_approach, percentage_error_aalst, percentage_error_hmm);

        p.difference_obs_predicted_allst = difference_obs_predicted_allst;
        p.difference_obs_predicted_approach = difference_obs_predicted_approach;
        p.difference_obs_predicted_hmm = difference_obs_predicted_hmm;

        xmm_predicted_data[product][state].push_back(p);
    }

}

void predictor::create_randomized_log()
{
    map<string, map<string,data::trace>> current_product_traces = d->product_traces;

    bool end_traces = false;
    while(!end_traces)
    {
        map<string, map<string,data::trace>>::iterator product_it = current_product_traces.begin();
        end_traces = true;
        while(product_it != current_product_traces.end()) 
        {
            if(product_it->second.size() != 0)
            {
                int selected_trace_index = rg.random_int(product_it->second.size()-1);
                map<string,data::trace>::iterator selected_trace = product_it->second.begin();
                std::advance(selected_trace, selected_trace_index);
                v_randomized_traces.push_back(selected_trace->second);
                current_product_traces[product_it->first].erase(selected_trace);
                end_traces = false;
            }
            product_it ++;
        }
    }
}
string predictor::create_prediction_state_string(double &real_time, data::trace &t, int string_size)
{
    int current_string_size = 0;
    string state = "";

    list<data::event>::iterator it = t.events.begin();
    real_time = t.events.back().d2 - it->d1;

    while(current_string_size < string_size)
    {
      real_time = t.events.back().d2 - it->d2;
      state += it->activity;
      current_string_size++;
      it++;
    }

    return state;
}

void predictor::generates_artificial_log()
{
    int sec_increment = 30;
    date d1("2019/02/02 23:59:01.000");
    for(int i=0;i<10;i++)
    {
        d1.print_date();
        d1.set_second(d1.get_second() + sec_increment);
    }
}

void predictor::calculate_errors(map<string, map<string, vector<predictor::predicted_values>>> &xmm_predicted_data, map<string, map<string, predictor::estimate_errors> > &xm_errors)
{
    for(map<string, map<string, vector<predictor::predicted_values>>>::iterator it = xmm_predicted_data.begin(); it != xmm_predicted_data.end(); it++)
    {
        for(map<string, vector<predictor::predicted_values>>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++)
        {

            double allst_percentage_error_sum = 0;
            double approach_percentage_error_sum = 0;
            double hmm_percentage_error_sum = 0;

            double allst_error_sum = 0;
            double approach_error_sum = 0;
            double hmm_error_sum = 0;

            double allst_error_2_pot_sum = 0;
            double approach_error_2_pot_sum = 0;
            double hmm_error_2_pot_sum = 0;

            for(size_t i = 0; i < it2->second.size(); i++)
            {
                allst_percentage_error_sum += it2->second[i].percentage_error_aalst;
                allst_error_sum += it2->second[i].difference_obs_predicted_allst;
                allst_error_2_pot_sum = allst_error_2_pot_sum + pow(it2->second[i].occured - it2->second[i].predicted_allst, 2);

                approach_percentage_error_sum += it2->second[i].percentage_error_approach;
                approach_error_sum += it2->second[i].difference_obs_predicted_approach;
                approach_error_2_pot_sum = approach_error_2_pot_sum + pow(it2->second[i].occured - it2->second[i].predicted_approach,2);

                hmm_percentage_error_sum += it2->second[i].percentage_error_hmm;
                hmm_error_sum += it2->second[i].difference_obs_predicted_hmm;
                hmm_error_2_pot_sum  = hmm_error_2_pot_sum + pow(it2->second[i].occured - it2->second[i].predicted_hmm, 2);
            }
            estimate_errors error;
            error.MAE_aalst     = allst_error_sum/it2->second.size();
            error.MAE_approach  = approach_error_sum/it2->second.size();
            error.MAE_hmm       = hmm_error_sum/it2->second.size();

            error.MAPE_aalst    = allst_percentage_error_sum/it2->second.size();
            error.MAPE_approach = approach_percentage_error_sum/it2->second.size();
            error.MAPE_hmm      = hmm_percentage_error_sum/it2->second.size();

            error.RMSE_aalst    = sqrt(allst_error_2_pot_sum/it2->second.size());
            error.RMSE_approach = sqrt(approach_error_2_pot_sum/it2->second.size());
            error.RMSE_hmm      = sqrt(hmm_error_2_pot_sum/it2->second.size());

            xm_errors[it->first][it2->first] = error;
        }
        for(map<string, vector<predictor::predicted_values>>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++)
        {
            double allst_st_deviation_sum = 0;
            double approach_st_deviation_sum = 0;
            double hmm_st_deviation_sum = 0;
            for(size_t i = 0; i < it2->second.size(); i++)
            {
                allst_st_deviation_sum    += sqrt(pow(it2->second[i].percentage_error_aalst - xm_errors[it->first][it2->first].MAPE_aalst,2));
                approach_st_deviation_sum += sqrt(pow(it2->second[i].percentage_error_approach - xm_errors[it->first][it2->first].MAPE_approach,2));
                hmm_st_deviation_sum      += sqrt(pow(it2->second[i].percentage_error_hmm - xm_errors[it->first][it2->first].MAPE_hmm,2));
            }
            xm_errors[it->first][it2->first].st_deviation_allst    = allst_st_deviation_sum/it2->second.size();
            xm_errors[it->first][it2->first].st_deviation_approach = approach_st_deviation_sum/it2->second.size();
            xm_errors[it->first][it2->first].st_deviation_hmm      = hmm_st_deviation_sum/it2->second.size();
        }
    }

}
void predictor::print_errors(string xfile_path, map<string, map<string, predictor::estimate_errors> > &xm_errors)
{
    ofstream o(xfile_path.c_str());
    o << "Product;State;MAPE_Approach;st_deviation_approach;MAPE_Allst;st_deviation_Allst;MAPE_hmm;st_deviation_hmm;MAE_approach;MAE_Allst;MAE_hmm;RMSE_Approach;RMSE_Aalst;RMSE_hmm\n";
    if(o.is_open())
    {
       for(map<string, map<string, predictor::estimate_errors>>::iterator it1 = xm_errors.begin(); it1 != xm_errors.end() ; it1++)
       {
           for(map<string, predictor::estimate_errors>::iterator it2 = it1->second.begin(); it2 != it1->second.end(); it2++)
           {
               o << it1->first << ";" << it2->first << ";"
                 << it2->second.MAPE_approach << ";" << it2->second.st_deviation_approach  << ";"
                 << it2->second.MAPE_aalst << ";" << it2->second.st_deviation_allst << ";"
                 << it2->second.MAPE_hmm << ";" << it2->second.st_deviation_hmm << ";"
                 << it2->second.MAE_approach << ";" << it2->second.MAE_aalst << ";" << it2->second.MAE_hmm << ";"
                 << it2->second.RMSE_approach << ";" << it2->second.RMSE_aalst << ";" << it2->second.RMSE_hmm << "\n";
           }
       }
    }
    o.close();
}

void predictor::initialize_markov()
{
    create_states_map();
    create_observation_matrix();
    create_state_from_to_matrix();
    initialize_markov_parameters();
}

void predictor::create_states_map()
{
    for(int i = 0; i < v_randomized_traces.size(); i++)
    {
        for(list<data::event>::iterator it = v_randomized_traces[i].events.begin(); it != v_randomized_traces[i].events.end(); it++)
        {
            if(m_states.find(it->activity) == m_states.end())
            {
                m_states[it->activity] = m_states.size();
                m_int_states[m_int_states.size()] = it->activity;
            }
        }
    }

}
void predictor::create_observation_matrix()
{
    int n_observations = v_randomized_traces.size();
    int n_observation_vector_size = m_states.size();

    xO = vector<vector<int>>(n_observations, vector<int>(n_observation_vector_size, 0));

    for(int i = 0; i < v_randomized_traces.size(); i++)
    {
        for(list<data::event>::iterator it = v_randomized_traces[i].events.begin(); it != v_randomized_traces[i].events.end(); it++)
        {
            xO[i][m_states[it->activity]] = 1;
        }
    }

}
void predictor::initialize_markov_parameters()
{

    A = vector<vector<double>>(m_states.size(), vector<double>(m_states.size(), 0));
    B = vector<vector<double>>(m_states.size(), vector<double>(2)); 
    pi = vector<double>(m_states.size());

    double sum_pi = 0;
    for(int i = 0; i < A.size(); i++)
    {
        double sum_A_line = 0;
        for(int j = 0; j < A[i].size(); j++)
        {
            int ran = rand() % 100;
            A[i][j] = ran;
            sum_A_line += ran;
        }
        for(int j = 0; j < A[i].size(); j++)
        {
            A[i][j] = A[i][j] / sum_A_line;
        }

        B[i][0] = rand() / double(RAND_MAX);
        B[i][1] = 1 - B[i][0];

        int ran = rand() % 100;
        pi[i] = ran;
        sum_pi += ran;

    }

    for(int i = 0; i < A.size(); i++)
    {
       pi[i] = pi[i] / sum_pi;
    }
}

double predictor::alpha(int xt, int xi, vector<int> O)
{
    vector<vector<double>> alpha_structure = vector<vector<double>>(xt + 1, vector<double>(m_states.size(), 0));

    for(int i = 0; i < pi.size(); i++)
    {
        alpha_structure[0][i] = pi[i] * B[i][O[i]];
    }

    if(xt > 0)
    {
        for(int t = 0; t < xt; t++) 
        {
            for(int j = 0; j < m_states.size(); j++)
            {
                double sum = 0;
                for(int i = 0; i < m_states.size(); i++ )
                {
                    sum += alpha_structure[t][i] * A[i][j];
                }
                alpha_structure[t + 1][j] = sum * B[j][O[t + 1]];
            }
        }
    }
    return alpha_structure[xt][xi];
}
double predictor::P_observation(vector<int> O)
{
    vector<vector<double>> alpha_structure = vector<vector<double>>(O.size(), vector<double>(m_states.size(), 0));
    for(int i = 0; i < pi.size(); i++)
    {
        alpha_structure[0][i] = pi[i] * B[i][O[i]];
    }
    for(int t = 0; t < O.size() - 1; t++) 
    {
        for(int j = 0; j < m_states.size(); j++)
        {
            double sum = 0;
            for(int i = 0; i < m_states.size(); i++ ) 
            {
                sum += alpha_structure[t][i] * A[i][j];
            }
            alpha_structure[t + 1][j] = sum * B[j][O[t + 1]];
        }
    }
    double probability = 0;
    for(int i = 0; i < m_states.size(); i++)
    {
        probability += alpha_structure[O.size() - 1][i];
    }

    return probability;

}

double predictor::betha(int xt, int xj, vector<int> O)
{
    vector<vector<double>> betha_structure = vector<vector<double>>(O.size(), vector<double>(m_states.size(), 0));
    for(int i = 0; i < pi.size(); i++)
    {
        betha_structure[O.size() - 1][i] = 1 ;
    }
    for(int t = O.size() - 2; t > -1; t--) 
    {
        for(int i = 0; i < m_states.size(); i++)
        {
            double sum = 0;
            for(int j = 0; j < m_states.size(); j++ ) 
            {
                sum += A[i][j] * B[j][O[t + 1]] * betha_structure[t + 1][j];
            }
            betha_structure[t][i] = sum;
        }
    }
    return betha_structure[xt][xj];
}
double predictor::epsilon(int xt, int xi, int xj, vector<int> O)
{
    double alpha_value = alpha(xt, xi, O);
    double betha_value = betha(xt + 1, xj, O);
    double numerator = alpha_value * B[xj][O[xt + 1]] * betha_value;
    double p_O = 0;

    for(int i = 0; i < m_states.size(); i++)
    {
        for(int j = 0; j < m_states.size(); j++)
        {
            p_O += alpha(xt, xi, O) * A[i][j] * B[j][O[xt + 1]] * betha(xt + 1, j, O);
        }
    }
    double epsilon_value = numerator / p_O;
    return epsilon_value;
}
double predictor::gamma(int xt, int xi, vector<int> O)
{
    double gamma_value = 0;
    for(int j = 0; j < m_states.size(); j++)
    {
        gamma_value += epsilon(xt, xi, j, O);
    }
    return  gamma_value;

}
double predictor::gamma_sum(int xi, vector<int> O)
{
    double gamma_sum_value = 0;
    for(int t = 0 ; t < O.size() - 1 ; t++)
    {
        gamma_sum_value += gamma(t, xi, O);
    }
    return  gamma_sum_value;
}
double predictor::epsilon_sum(int xi, int xj, vector<int> O)
{
    double epsilon_sum_value = 0;
    for(int t = 0 ; t < O.size() - 1 ; t++)
    {
        epsilon_sum_value += epsilon(t, xi, xj, O);
    }
    return  epsilon_sum_value;
}
void predictor::estimate_markov_model_parameters(int training_data_size, int convergence_limit)
{
    vector<double> new_pi;
    vector<vector<double>> new_A;
    vector<vector<double>> new_B;
    stopwatch st;
    st.new_stopwatch("Estimation");
    st.start("Estimation");

    for(int i = 0; i < convergence_limit; i++)
    {
        print_parameters(i, st.ellapsed_seconds("Estimation"));
        cout << "Convergence " << i << "\n Estimating A\n";
        new_A = estimate_A(training_data_size);
        cout << "Estimating B\n";
        new_B = estimate_B(training_data_size);
        cout << "Estimating pi\n";
        new_pi = estimate_pi(training_data_size);
        A = new_A;
        B = new_B;
        pi = new_pi;
    }
    st.stop("Estimation");
}

vector<double> predictor::estimate_pi(int training_data_size)
{
    int limit = xO.size() > training_data_size?training_data_size:xO.size();

    vector<double> new_pi(pi.size(), 0);

    for(int o = 0; o < limit; o++) 
    {
        for(int i = 0; i < new_pi.size(); i++)
        {
            new_pi[i] += gamma(0, i, xO[o]);
        }
        cout << "pi - obs. " << o << "\n";
    }

    double sum = 0;
    for(int i = 0; i < new_pi.size(); i++)
    {
        sum += new_pi[i] / limit;
        new_pi[i] = new_pi[i] / limit;
    }
    for(int i = 0; i < new_pi.size(); i++)
    {
        new_pi[i] = new_pi[i] / sum;
    }
    return new_pi;
}

vector<vector<double> > predictor::estimate_A(int training_data_size)
{

    int limit = xO.size() > training_data_size?training_data_size:xO.size();
    vector<vector<double>> new_A = vector<vector<double>>(m_states.size(), vector<double>(m_states.size(), 0));
    for(int i = 0; i < new_A.size(); i++)
    {
        double line_sum = 0;
        double sum_n_gamma = 0;
        for(int k = 0; k < limit; k++)
        {
            sum_n_gamma += gamma_sum(i, xO[k]);
        }
        double sum_gamma = sum_n_gamma / limit;

        for(int j = 0; j < new_A[i].size(); j++)
        {
           double sum_n_epsilon = 0;
           for(int l = 0; l < limit; l++)
            {
                sum_n_epsilon += epsilon_sum(i, j, xO[l]);
            }
           double sum_epsilon = sum_n_epsilon / limit;
           new_A[i][j] = sum_epsilon / sum_gamma;
           line_sum += new_A[i][j];
           cout << "A[" << i << "," << j << "] \n";
        }
        for(int j = 0; j < new_A[i].size(); j++)
        {
           new_A[i][j] = new_A[i][j] / line_sum;
        }
    }

    return new_A;
}

vector<vector<double> > predictor::estimate_B(int training_data_size)
{
    int limit = xO.size() > training_data_size?training_data_size:xO.size();
    vector<vector<double>> new_B = vector<vector<double>>(m_states.size(), vector<double>(2));
    for(int j = 0; j < B.size(); j++)
    {
        double line_sum = 0;
        for(int k = 0; k < 2; k++)
        {
            double gamma_sum_num = 0;
            double gamma_sum_den = 0;
            double current_gamma = 0;
            for(int o = 0; o < limit; o++)
            {
                for(int t = 0 ; t < xO[o].size() - 1 ; t++)
                {
                    current_gamma = gamma(t, j, xO[o]);
                    if(xO[o][t] == k)
                    {
                        gamma_sum_num += current_gamma;
                    }
                    gamma_sum_den += current_gamma;
                }
            }
            gamma_sum_den = gamma_sum_den / limit;
            gamma_sum_num = gamma_sum_num / limit;
            new_B[j][k] = gamma_sum_num / gamma_sum_den;
            line_sum += new_B[j][k];
            cout << "B[" << j << "," << k << "] \n";
        }
        for(int k = 0; k < 2; k++)
        {
            new_B[j][k] = new_B[j][k] / line_sum;
        }    }
    return  new_B;
}

void predictor::print_parameters(int i, double ellapsed_time)
{
    string path = "MarkovParameters/E_" + std::to_string(i) + ".txt";

    ofstream o;
    o.open(path.c_str());

    o << "Ellapsed time " << ellapsed_time << "\n";
    o << "pi = \n";
    for(int i = 0; i < pi.size(); i++)
    {
        o << pi[i] << " ";
    }
    o << "\n";

    o << "A = \n";
    for(int i = 0; i < A.size(); i++)
    {
        for(int j = 0; j < A.size(); j++)
        {
            o << A[i][j] << " ";
        }
        o << " \n";
    }
    o << "\n";

    o << "B = \n";
    for(int i = 0; i < B.size(); i++)
    {
        for(int j = 0; j < B[i].size(); j++)
        {
            o << B[i][j] << " ";
        }
        o << " \n";
    }
    o.close();
}

void predictor::read_markov_parameters(string parameters_file)
{
    int line = 0;
    int column = 0;

    ifstream reader(parameters_file.c_str());

    reader >> line;
    reader >> column;
    for(int i = 0; i < column; i++)
    {
        reader >> pi[i];
    }

    reader >> line;
    reader >> column;
    for(int i = 0; i < line; i++)
    {
        for(int j = 0; j < column; j++)
        {
            reader >> A[i][j];
        }
    }

    reader >> line;
    reader >> column;
    for(int i = 0; i < line; i++)
    {
        for(int j = 0; j < column; j++)
        {
            reader >> B[i][j];
        }
    }
}
void predictor::create_state_from_to_matrix()
{
    state_from_to_matrix = vector<vector<int>>(m_states.size(), vector<int>(m_states.size()));
    for(vector<data::trace>::iterator it = v_randomized_traces.begin(); it != v_randomized_traces.end(); it++)
    {
        bool stop_criteria = false;
        list<data::event>::iterator it2 = it->events.begin();
        while(!stop_criteria)
        {
            string stat_i = it2->activity;
            string stat_j;
            it2++;
            if(it2 == it->events.end())
            {
                stop_criteria = true;
            }else
            {
                 stat_j = it2->activity;
                 state_from_to_matrix[m_states[stat_i]][m_states[stat_j]] = 1;
            }
        }
    }
}

double predictor::predict_markov_value(data::trace &t, int string_size, string prediction_string)
{
    return 0;
    if(prediction_string == "")
     return d->complete_transition_system[prediction_string].mean;

    if(string_size == t.events.size() - 1)
        return d->complete_transition_system[prediction_string].mean;

    double markov_predicted_value = 0;
    string prediction_state = "";
    list<data::event>::iterator it = t.events.begin();
    int current_string_size = 0;
    string last_state = "";
    vector<int> HMM_state(m_states.size(), 0);
    while(current_string_size < string_size)
    {
      HMM_state[m_states[it->activity]] = 1;
      prediction_state += it->activity;
      last_state = it->activity;
      current_string_size ++;
      it ++;
    }
    vector<string> possible_next_states;
    for(int j = 0; j < state_from_to_matrix.size(); j++)
    {
        if(state_from_to_matrix[m_states[last_state]][j] == 1)
        {
            possible_next_states.push_back(m_int_states[j]);
        }
    }
    if(possible_next_states.size() == 0)
    {
        markov_predicted_value = d->complete_transition_system[prediction_state].mean;
    }else
    {
        vector<vector<int>> HMM_next_possible_states = vector<vector<int>>(possible_next_states.size(), vector<int>{ HMM_state});
        vector<double> P_o_values = vector<double>(possible_next_states.size());
        vector<double> ATS_estimation = vector<double>(possible_next_states.size());
        double sum_P_o_times_ATS_estimate = 0;
        double sum_P_o = 0;
        for(int i = 0; i < possible_next_states.size(); i++)
        {
            HMM_next_possible_states[i][m_states[possible_next_states[i]]] = 1;
            double P_o = P_observation(HMM_next_possible_states[i]);
            sum_P_o += P_o;
            string ATS_next_state = prediction_state + possible_next_states[i];
            sum_P_o_times_ATS_estimate += P_o * d->complete_transition_system[ATS_next_state].mean;
        }
        markov_predicted_value = sum_P_o_times_ATS_estimate / sum_P_o;
    }

    return markov_predicted_value;
}

