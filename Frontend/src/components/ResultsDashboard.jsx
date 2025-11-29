import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  CheckCircle, AlertTriangle, XCircle, Download, 
  Briefcase, TrendingUp, BookOpen, User, DollarSign,
  ChevronDown, ChevronUp, FileText
} from 'lucide-react';
import jsPDF from 'jspdf';
import 'jspdf-autotable';

const Card = ({ children, className = "" }) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
    className={`bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden ${className}`}
  >
    {children}
  </motion.div>
);

const SectionHeader = ({ icon: Icon, title, color = "text-gray-800" }) => (
  <div className="flex items-center gap-2 mb-4 border-b border-gray-100 pb-3">
    <Icon className={`w-6 h-6 ${color}`} />
    <h2 className={`text-xl font-bold ${color}`}>{title}</h2>
  </div>
);

const ScoreRing = ({ score, label }) => {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  
  const getColor = (s) => {
    if (s >= 80) return '#10B981'; // Green
    if (s >= 60) return '#F59E0B'; // Yellow
    return '#EF4444'; // Red
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke="#E5E7EB"
            strokeWidth="8"
            fill="transparent"
          />
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke={getColor(score)}
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
          <span className="text-3xl font-bold text-gray-800">{score}</span>
        </div>
      </div>
      <span className="mt-2 font-medium text-gray-600">{label}</span>
    </div>
  );
};

const ResultsDashboard = ({ results, resumeText, jobDescription }) => {
  if (!results) return null;
  const { atsAnalyzer, atsOptimizer, interviewCoach } = results;
  const [generatingPDF, setGeneratingPDF] = useState(false);

  const generateAnswersPDF = async () => {
    if (!interviewCoach || !interviewCoach.questions) return;
    
    setGeneratingPDF(true);
    try {
      // 1. Fetch answers from backend
      const response = await fetch('http://localhost:8000/generate-answers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resumeText,
          jobDescription,
          questions: interviewCoach.questions
        })
      });

      if (!response.ok) throw new Error("Failed to generate answers");
      const data = await response.json();
      const answers = data.answers;

      // 2. Generate PDF
      const doc = new jsPDF();
      
      // Title
      doc.setFontSize(20);
      doc.setTextColor(75, 108, 183); // Primary color
      doc.text("Interview Preparation Guide", 14, 22);
      
      doc.setFontSize(12);
      doc.setTextColor(100);
      doc.text(`Target Role: ${interviewCoach.targetRole || "General"}`, 14, 32);
      
      let yPos = 40;

      answers.forEach((item, index) => {
        // Check page break
        if (yPos > 250) {
          doc.addPage();
          yPos = 20;
        }

        // Question
        doc.setFontSize(12);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(0);
        const questionLines = doc.splitTextToSize(`Q${index + 1}: ${item.question}`, 180);
        doc.text(questionLines, 14, yPos);
        yPos += questionLines.length * 7;

        // Answer
        doc.setFontSize(11);
        doc.setFont("helvetica", "normal");
        doc.setTextColor(50);
        const answerLines = doc.splitTextToSize(item.answer, 180);
        doc.text(answerLines, 14, yPos);
        yPos += answerLines.length * 6 + 10; // Spacing
      });

      doc.save("NeuraResume_Interview_Prep.pdf");

    } catch (error) {
      console.error("PDF Generation Error:", error);
      alert("Failed to generate PDF. Please try again.");
    } finally {
      setGeneratingPDF(false);
    }
  };

  return (
    <div className="space-y-8 pb-12">
      
      {/* 1. ATS Analysis & Suitability */}
      {atsAnalyzer && (
        <Card className="p-6">
          <SectionHeader icon={FileText} title="Resume Analysis" color="text-primary" />
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Score */}
            <div className="flex justify-center">
              <ScoreRing score={atsAnalyzer.atsScore?.score || 0} label="ATS Score" />
            </div>

            {/* Key Insights */}
            <div className="md:col-span-2 space-y-6">
              {/* Career Fit & Suitability */}
              <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                <h3 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
                  <Briefcase size={18} className="text-indigo-600" />
                  Career Fit & Insights
                </h3>
                
                {/* Job Match (if available) */}
                {atsAnalyzer.jobSuitability && (
                  <div className={`mb-4 p-3 rounded-lg border-l-4 ${
                    atsAnalyzer.jobSuitability.match === 'High' ? 'bg-green-50 border-green-500' :
                    atsAnalyzer.jobSuitability.match === 'Medium' ? 'bg-yellow-50 border-yellow-500' :
                    'bg-red-50 border-red-500'
                  }`}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-bold text-gray-800">Match: {atsAnalyzer.jobSuitability.match}</span>
                      <span className="font-bold">{atsAnalyzer.jobSuitability.percentage}%</span>
                    </div>
                    <p className="text-xs text-gray-600 leading-relaxed">{atsAnalyzer.jobSuitability.reasoning}</p>
                  </div>
                )}

                {/* Recommended Roles */}
                {atsAnalyzer.careerSuggestions && (
                  <div className="space-y-3">
                    <div>
                      <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Recommended Roles</h4>
                      <div className="flex flex-wrap gap-2">
                        {atsAnalyzer.careerSuggestions.recommendedRoles.map((role, idx) => (
                          <span key={idx} className="px-2 py-1 bg-indigo-50 text-indigo-700 text-xs font-medium rounded border border-indigo-100">
                            {role}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">Market Outlook</h4>
                      <p className="text-sm text-gray-600">{atsAnalyzer.careerSuggestions.marketOutlook}</p>
                    </div>

                    {atsAnalyzer.careerSuggestions.topCompaniesToTarget && (
                      <div>
                        <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">Target Companies</h4>
                        <div className="flex flex-wrap gap-2">
                          {atsAnalyzer.careerSuggestions.topCompaniesToTarget.map((company, idx) => (
                            <span key={idx} className="text-xs text-gray-600 bg-gray-100 px-2 py-0.5 rounded">
                              {company}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Persona & Salary */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {atsAnalyzer.resumePersona && (
                  <div className="bg-indigo-50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2 text-indigo-700">
                      <User size={18} />
                      <span className="font-bold">Resume Persona</span>
                    </div>
                    <p className="font-semibold text-gray-800">{atsAnalyzer.resumePersona.tone}</p>
                    <p className="text-xs text-gray-600 mt-1">{atsAnalyzer.resumePersona.impression}</p>
                  </div>
                )}
                
                {atsAnalyzer.salaryEstimation && (
                  <div className="bg-green-50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2 text-green-700">
                      <DollarSign size={18} />
                      <span className="font-bold">Est. Salary</span>
                    </div>
                    <p className="font-semibold text-gray-800">{atsAnalyzer.salaryEstimation.range}</p>
                    <p className="text-xs text-gray-600 mt-1">{atsAnalyzer.salaryEstimation.currency}</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Strengths & Weaknesses */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8 pt-6 border-t border-gray-100">
            <div>
              <h4 className="font-semibold text-green-700 mb-3 flex items-center gap-2">
                <CheckCircle size={18} /> Strengths
              </h4>
              <ul className="space-y-2">
                {atsAnalyzer.strengths?.map((s, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0"></span>
                    {s}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-red-700 mb-3 flex items-center gap-2">
                <AlertTriangle size={18} /> Areas for Improvement
              </h4>
              <ul className="space-y-2">
                {atsAnalyzer.weaknesses?.map((w, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0"></span>
                    {w}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      )}

      {/* 2. Optimization & Learning Path */}
      {atsOptimizer && (
        <Card className="p-6">
          <SectionHeader icon={TrendingUp} title="Optimization Plan" color="text-blue-700" />
          
          <div className="mb-6">
            <h3 className="font-semibold text-gray-800 mb-2">Strategic Overview</h3>
            <p className="text-gray-600 bg-blue-50 p-4 rounded-lg border border-blue-100">
              {atsOptimizer.overallStrategy}
            </p>
          </div>

          {/* Skill Gap Learning Path */}
          {atsOptimizer.skillGapLearningPath && atsOptimizer.skillGapLearningPath.length > 0 && (
            <div className="mb-8">
              <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <BookOpen size={18} className="text-purple-600" />
                Skill Gap Learning Path
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {atsOptimizer.skillGapLearningPath.map((item, idx) => (
                  <div key={idx} className="border border-purple-100 bg-purple-50 rounded-lg p-4">
                    <p className="font-bold text-purple-800 mb-2">{item.skill}</p>
                    <div className="flex flex-wrap gap-2">
                      {item.learningTopics.map((topic, tIdx) => (
                        <span key={tIdx} className="px-2 py-1 bg-white text-purple-600 text-xs rounded border border-purple-200">
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-4">
            <h3 className="font-semibold text-gray-800">Detailed Suggestions</h3>
            {atsOptimizer.sectionLevelSuggestions?.map((suggestion, idx) => (
              <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow bg-white">
                <div className="flex justify-between items-start mb-2">
                  <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs font-bold rounded uppercase">
                    {suggestion.section}
                  </span>
                </div>
                <p className="text-red-600 text-sm mb-2 font-medium">Problem: {suggestion.issue}</p>
                <p className="text-green-700 text-sm mb-3 font-medium">Fix: {suggestion.suggestion}</p>
                {suggestion.exampleRewrite && (
                  <div className="bg-gray-50 p-3 rounded text-xs font-mono text-gray-600 border border-gray-100">
                    "{suggestion.exampleRewrite}"
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* 3. Interview Coach */}
      {interviewCoach && (
        <Card className="p-6">
          <div className="flex justify-between items-center mb-6 border-b border-gray-100 pb-3">
            <div className="flex items-center gap-2">
              <Briefcase className="w-6 h-6 text-indigo-700" />
              <h2 className="text-xl font-bold text-indigo-700">Interview Coach</h2>
            </div>
            <button
              onClick={generateAnswersPDF}
              disabled={generatingPDF}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-white font-medium transition-all ${
                generatingPDF 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-indigo-600 hover:bg-indigo-700 shadow-md hover:shadow-lg'
              }`}
            >
              {generatingPDF ? (
                <>Generating...</>
              ) : (
                <>
                  <Download size={18} />
                  Download Answers PDF
                </>
              )}
            </button>
          </div>

          <div className="space-y-4">
            {interviewCoach.questions?.map((q, idx) => (
              <div key={idx} className="group border border-gray-100 rounded-lg p-4 hover:border-indigo-200 hover:bg-indigo-50/30 transition-colors">
                <div className="flex justify-between items-start mb-2">
                  <span className={`text-xs font-bold px-2 py-1 rounded uppercase ${
                    q.difficulty === 'Easy' ? 'bg-green-100 text-green-800' :
                    q.difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {q.difficulty}
                  </span>
                  <span className="text-xs text-gray-400">{q.category}</span>
                </div>
                <p className="font-medium text-gray-800 mb-2">{q.question}</p>
                {q.followUpHint && (
                  <div className="flex items-start gap-2 text-sm text-gray-500 italic">
                    <span className="font-semibold not-italic text-indigo-400">Hint:</span>
                    {q.followUpHint}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default ResultsDashboard;
