import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, CheckCircle, Sparkles, Brain, Target } from 'lucide-react';
import ResultsDashboard from './components/ResultsDashboard';

function App() {
  const [resumeText, setResumeText] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [tasks, setTasks] = useState({
    runAtsAnalyzer: true,
    runAtsOptimizer: true,
    runInterviewCoach: true
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!resumeText.trim() && !resumeFile) {
      setError("Please provide resume text or upload a PDF file.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      if (resumeText) formData.append('resumeText', resumeText);
      if (resumeFile) formData.append('resumeFile', resumeFile);
      if (jobDescription) formData.append('jobDescription', jobDescription);
      formData.append('tasks', JSON.stringify(tasks));

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error: ${response.statusText}`);
      }

      const text = await response.text();
      try {
        const data = JSON.parse(text);
        setResults(data);
      } catch (e) {
        console.error("Failed to parse JSON:", text);
        throw new Error("Received invalid JSON from backend. Check console for details.");
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900 selection:bg-indigo-100 selection:text-indigo-900">
      
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center text-white shadow-lg shadow-indigo-200">
              <Brain size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 tracking-tight">NeuraResume</h1>
              <p className="text-xs text-gray-500 font-medium">AI-Powered Career Architect</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-6 text-sm font-medium text-gray-600">
            <span className="flex items-center gap-1 hover:text-indigo-600 cursor-pointer transition"><Target size={16}/> ATS Analysis</span>
            <span className="flex items-center gap-1 hover:text-indigo-600 cursor-pointer transition"><Sparkles size={16}/> Optimization</span>
            <span className="flex items-center gap-1 hover:text-indigo-600 cursor-pointer transition"><FileText size={16}/> Interview Prep</span>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Input Section */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-4 space-y-6"
          >
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 sticky top-24">
              <h2 className="text-lg font-bold mb-6 text-gray-800 flex items-center gap-2">
                <Upload size={20} className="text-indigo-600" />
                Input Details
              </h2>
              
              {/* Resume Input */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-700 mb-2">Resume</label>
                <div className="space-y-3">
                  <div className="relative group">
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => setResumeFile(e.target.files[0])}
                      className="block w-full text-sm text-gray-500 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 transition cursor-pointer border border-gray-200 rounded-lg"
                    />
                  </div>
                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-gray-200"></div>
                    </div>
                    <div className="relative flex justify-center text-xs uppercase">
                      <span className="bg-white px-2 text-gray-400">Or paste text</span>
                    </div>
                  </div>
                  <textarea
                    className="w-full h-32 p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm transition bg-gray-50 focus:bg-white"
                    placeholder="Paste resume text here..."
                    value={resumeText}
                    onChange={(e) => setResumeText(e.target.value)}
                  />
                </div>
              </div>

              {/* Job Description */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-700 mb-2">Target Job Description (Optional)</label>
                <textarea
                  className="w-full h-32 p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm transition bg-gray-50 focus:bg-white"
                  placeholder="Paste the job description here for tailored analysis..."
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                />
              </div>

              {/* Agent Selection */}
              <div className="mb-8">
                <label className="block text-sm font-semibold text-gray-700 mb-3">Analysis Modules</label>
                <div className="space-y-3">
                  {[
                    { id: 'runAtsAnalyzer', label: 'ATS Analysis & Suitability', desc: 'Score, keywords, and persona' },
                    { id: 'runAtsOptimizer', label: 'Optimization & Upskilling', desc: 'Improvements and learning path' },
                    { id: 'runInterviewCoach', label: 'Interview Coach', desc: 'Questions and model answers' }
                  ].map((agent) => (
                    <label key={agent.id} className="flex items-start gap-3 p-3 rounded-lg border border-gray-100 hover:bg-gray-50 cursor-pointer transition group">
                      <div className="relative flex items-center pt-1">
                        <input
                          type="checkbox"
                          checked={tasks[agent.id]}
                          onChange={(e) => setTasks({...tasks, [agent.id]: e.target.checked})}
                          className="w-4 h-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                        />
                      </div>
                      <div>
                        <span className="block text-sm font-medium text-gray-900 group-hover:text-indigo-700 transition">{agent.label}</span>
                        <span className="block text-xs text-gray-500">{agent.desc}</span>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              <button
                onClick={handleAnalyze}
                disabled={loading}
                className={`w-full py-3.5 px-4 rounded-xl text-white font-bold shadow-lg shadow-indigo-200 transition-all transform active:scale-95 ${
                  loading
                    ? 'bg-indigo-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700'
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing Profile...
                  </span>
                ) : (
                  'Run AI Analysis'
                )}
              </button>
              
              {error && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 bg-red-50 text-red-700 text-sm rounded-lg border border-red-100 flex items-start gap-2"
                >
                  <div className="mt-0.5">⚠️</div>
                  <div>{error}</div>
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* Results Section */}
          <div className="lg:col-span-8">
            {results ? (
              <ResultsDashboard results={results} resumeText={resumeText} jobDescription={jobDescription} />
            ) : (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="h-full flex flex-col items-center justify-center text-center p-12 bg-white rounded-xl border border-dashed border-gray-300 min-h-[600px]"
              >
                <div className="w-24 h-24 bg-indigo-50 rounded-full flex items-center justify-center mb-6">
                  <Sparkles className="w-10 h-10 text-indigo-400" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-2">Ready to Optimize?</h3>
                <p className="text-gray-500 max-w-md mx-auto">
                  Upload your resume and a job description to unlock AI-powered insights, salary estimation, and tailored interview prep.
                </p>
              </motion.div>
            )}
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
