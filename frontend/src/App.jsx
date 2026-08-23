import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Upload,
  FileVideo,
  Info,
  Activity,
  AlertTriangle,
  Layers,
  Search,
  CheckCircle2,
  XCircle,
  HelpCircle,
  BarChart3,
  Sliders,
  Video,
  X
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

const API_BASE_URL = 'http://127.0.0.1:8000';

const FEATURE_FAMILY_COLORS = {
  'Color (HSV)': '#3b82f6',
  'Texture (LBP & Gray)': '#10b981',
  'Texture (GLCM)': '#8b5cf6',
  'Frequency (DCT)': '#f59e0b',
  'Edge (Canny)': '#ec4899',
  'Other': '#64748b'
};

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [showModelModal, setShowModelModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);

  // Fetch API Health and Model Info on startup
  useEffect(() => {
    fetchHealth();
    fetchModelInfo();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (e) {
      setHealth({ status: 'offline' });
    }
  };

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/model-info`);
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch (e) {
      console.error('Failed to fetch model info', e);
    }
  };

  const handleFileSelect = (file) => {
    if (!file) return;

    // Validate size (100MB max)
    if (file.size > 100 * 1024 * 1024) {
      setError('Selected video exceeds maximum allowed limit of 100 MB.');
      return;
    }

    // Validate extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp4', 'mov', 'avi', 'webm', 'mkv'].includes(ext)) {
      setError('Unsupported file format. Please select an MP4, MOV, AVI, or WEBM video.');
      return;
    }

    setError(null);
    setSelectedFile(file);
    setResult(null);

    // Create HTML5 video preview URL
    if (filePreviewUrl) {
      URL.revokeObjectURL(filePreviewUrl);
    }
    const url = URL.createObjectURL(file);
    setFilePreviewUrl(url);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setProgress(15);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const interval = setInterval(() => {
        setProgress((prev) => (prev < 85 ? prev + 10 : prev));
      }, 400);

      const res = await fetch(`${API_BASE_URL}/api/analyze?n_frames=10`, {
        method: 'POST',
        body: formData
      });

      clearInterval(interval);
      setProgress(100);

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Analysis request failed.');
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message || 'Error executing video analysis.');
    } finally {
      setAnalyzing(false);
    }
  };

  // Build combined table data if result exists
  const getTableData = () => {
    if (!result || !result.explainability) return [];

    const fakeEv = result.explainability.top_fake_evidence || [];
    const realEv = result.explainability.top_real_evidence || [];
    const combined = [...fakeEv, ...realEv];

    if (!searchTerm) return combined;

    return combined.filter(
      (item) =>
        item.feature.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.feature_family.toLowerCase().includes(searchTerm.toLowerCase())
    );
  };

  return (
    <div className="app-container">
      {/* HEADER SECTION */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1.25rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Activity style={{ color: '#3b82f6', width: '28px', height: '28px' }} />
            <h1 style={{ fontSize: '1.5rem', fontWeight: '700', letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
              Media Forensics Lab
            </h1>
            <span className="badge badge-prototype">Research Prototype</span>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
            Explainable Video Authenticity Analysis using Handcrafted Visual Features & SHAP Attribution
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {/* API Health Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8125rem', background: 'var(--bg-surface)', padding: '0.4rem 0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: health?.status === 'online' ? '#10b981' : '#ef4444' }} />
            <span style={{ color: 'var(--text-secondary)' }}>API: {health?.status === 'online' ? 'Online' : 'Offline'}</span>
          </div>

          <button className="btn-secondary" onClick={() => setShowModelModal(true)}>
            <Info size={16} />
            <span>Model Spec</span>
          </button>
        </div>
      </header>

      {/* RESEARCH PROTOTYPE DISCLAIMER BANNER */}
      <div style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', borderRadius: '8px', padding: '0.875rem 1.25rem', marginBottom: '2rem', display: 'flex', alignItems: 'flex-start', gap: '0.875rem' }}>
        <AlertTriangle style={{ color: '#f59e0b', width: '20px', height: '20px', flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.8125rem', color: '#fcd34d', lineHeight: '1.45' }}>
          <strong>Research Benchmark Disclaimer:</strong> This system is a forensic research prototype evaluating 216 traditional handcrafted visual features.
          Full dataset benchmark validation achieved <strong>Balanced Accuracy: 50.00%</strong> and <strong>ROC-AUC: 53.43%</strong>. Decision probabilities represent algorithmic decision estimates and must not be treated as ground-truth proof of video authenticity.
        </div>
      </div>

      {/* MAIN CONTENT GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: result ? '380px 1fr' : '1fr', gap: '1.75rem', alignItems: 'start' }}>

        {/* LEFT PANEL: UPLOAD & INPUT SECTION */}
        <div>
          <div className="forensic-card" style={{ marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '1rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Upload size={18} style={{ color: '#3b82f6' }} />
              <span>Video Input Workspace</span>
            </h2>

            {/* Dropzone */}
            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: '2px dashed var(--border-color)',
                borderRadius: '8px',
                padding: '2rem 1.5rem',
                textAlign: 'center',
                cursor: 'pointer',
                background: 'var(--bg-surface-elevated)',
                transition: 'all 0.15s ease',
                marginBottom: '1rem'
              }}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => handleFileSelect(e.target.files[0])}
                accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
                style={{ display: 'none' }}
              />
              <FileVideo size={36} style={{ color: 'var(--text-muted)', marginBottom: '0.75rem' }} />
              <p style={{ fontSize: '0.875rem', fontWeight: '500', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                Drag and drop MP4 video here
              </p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                or click to browse local files (max 100 MB)
              </p>
            </div>

            {/* Selected File Details */}
            {selectedFile && (
              <div style={{ background: 'var(--bg-primary)', padding: '0.75rem 1rem', borderRadius: '6px', border: '1px solid var(--border-color)', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', overflow: 'hidden' }}>
                  <Video size={18} style={{ color: '#3b82f6', flexShrink: 0 }} />
                  <div style={{ overflow: 'hidden' }}>
                    <p style={{ fontSize: '0.8125rem', fontWeight: '500', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {selectedFile.name}
                    </p>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedFile(null);
                    setResult(null);
                  }}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
                >
                  <X size={16} />
                </button>
              </div>
            )}

            {/* Analyze Button */}
            <button
              className="btn-primary"
              style={{ width: '100%', justifyContent: 'center' }}
              onClick={handleAnalyze}
              disabled={!selectedFile || analyzing}
            >
              {analyzing ? (
                <>
                  <Activity size={18} style={{ animation: 'spin 1s linear infinite' }} />
                  <span>Executing Pipeline... {progress}%</span>
                </>
              ) : (
                <>
                  <Sliders size={18} />
                  <span>Execute Forensic Analysis</span>
                </>
              )}
            </button>

            {/* Error Message */}
            {error && (
              <div style={{ marginTop: '1rem', background: 'var(--accent-fake-bg)', border: '1px solid var(--accent-fake-border)', padding: '0.75rem 1rem', borderRadius: '6px', fontSize: '0.8125rem', color: 'var(--accent-fake)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <XCircle size={16} style={{ flexShrink: 0 }} />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* HTML5 Video Player Preview */}
          {filePreviewUrl && (
            <div className="forensic-card">
              <h3 style={{ fontSize: '0.875rem', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '0.75rem' }}>
                Video Stream Preview
              </h3>
              <video
                ref={videoRef}
                src={filePreviewUrl}
                controls
                style={{ width: '100%', borderRadius: '6px', background: '#000', maxHeight: '220px' }}
              />
            </div>
          )}
        </div>

        {/* RIGHT PANEL: FORENSIC ANALYSIS DASHBOARD */}
        {result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

            {/* 1. VERDICT BANNER CARD */}
            <div className="forensic-card" style={{ borderLeft: result.prediction?.label === 'FAKE' ? '4px solid #ef4444' : '4px solid #10b981' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
                <div>
                  <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Algorithmic Verdict
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.25rem' }}>
                    {result.prediction?.label === 'FAKE' ? (
                      <ShieldAlert size={32} style={{ color: '#ef4444' }} />
                    ) : (
                      <ShieldCheck size={32} style={{ color: '#10b981' }} />
                    )}
                    <h2 style={{ fontSize: '2rem', fontWeight: '700', color: result.prediction?.label === 'FAKE' ? '#ef4444' : '#10b981' }}>
                      {result.prediction?.label}
                    </h2>
                  </div>
                </div>

                {/* Score Gauge */}
                <div style={{ background: 'var(--bg-primary)', padding: '0.875rem 1.25rem', borderRadius: '6px', border: '1px solid var(--border-color)', minWidth: '220px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                    <span>Model Decision Score</span>
                    <span className="font-mono">{(result.prediction?.fake_probability * 100).toFixed(1)}% FAKE</span>
                  </div>
                  <div style={{ height: '8px', background: '#334155', borderRadius: '4px', overflow: 'hidden' }}>
                    <div
                      style={{
                        height: '100%',
                        width: `${(result.prediction?.fake_probability * 100).toFixed(1)}%`,
                        background: result.prediction?.fake_probability > 0.5 ? '#ef4444' : '#10b981',
                        transition: 'width 0.5s ease'
                      }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
                    <span>REAL (0.0)</span>
                    <span>FAKE (1.0)</span>
                  </div>
                </div>
              </div>

              {/* Stream Stats Line */}
              <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color-subtle)', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', fontSize: '0.8125rem' }}>
                <div>
                  <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Total Frames</span>
                  <span className="font-mono" style={{ fontWeight: '600' }}>{result.video?.total_frames_in_stream}</span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Analyzed Frames</span>
                  <span className="font-mono" style={{ fontWeight: '600' }}>{result.video?.frames_analyzed}</span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Extracted Features</span>
                  <span className="font-mono" style={{ fontWeight: '600' }}>{result.video?.features_extracted}</span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Model Classifier</span>
                  <span className="font-mono" style={{ fontWeight: '600' }}>{result.model?.name}</span>
                </div>
              </div>
            </div>

            {/* 2. EXPLAINABILITY SHAP EVIDENCE SECTION */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              {/* TOP FAKE EVIDENCE */}
              <div className="forensic-card">
                <h3 style={{ fontSize: '0.9375rem', fontWeight: '600', color: '#ef4444', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ShieldAlert size={18} />
                  <span>Top Evidence Pushing FAKE</span>
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {result.explainability?.top_fake_evidence?.map((item, idx) => (
                    <div key={idx} style={{ background: 'var(--bg-primary)', padding: '0.75rem 1rem', borderRadius: '6px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <p className="font-mono" style={{ fontSize: '0.8125rem', fontWeight: '500', color: 'var(--text-primary)' }}>
                          {item.feature}
                        </p>
                        <span className="badge" style={{ background: 'rgba(59, 130, 246, 0.12)', color: '#3b82f6', marginTop: '0.25rem' }}>
                          {item.feature_family}
                        </span>
                      </div>
                      <span className="font-mono" style={{ fontSize: '0.875rem', fontWeight: '600', color: '#ef4444' }}>
                        +{item.shap_value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* TOP REAL EVIDENCE */}
              <div className="forensic-card">
                <h3 style={{ fontSize: '0.9375rem', fontWeight: '600', color: '#10b981', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ShieldCheck size={18} />
                  <span>Top Evidence Pushing REAL</span>
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {result.explainability?.top_real_evidence?.map((item, idx) => (
                    <div key={idx} style={{ background: 'var(--bg-primary)', padding: '0.75rem 1rem', borderRadius: '6px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <p className="font-mono" style={{ fontSize: '0.8125rem', fontWeight: '500', color: 'var(--text-primary)' }}>
                          {item.feature}
                        </p>
                        <span className="badge" style={{ background: 'rgba(59, 130, 246, 0.12)', color: '#3b82f6', marginTop: '0.25rem' }}>
                          {item.feature_family}
                        </span>
                      </div>
                      <span className="font-mono" style={{ fontSize: '0.875rem', fontWeight: '600', color: '#10b981' }}>
                        {item.shap_value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 3. FEATURE FAMILY CONTRIBUTION CHART */}
            <div className="forensic-card">
              <h3 style={{ fontSize: '0.9375rem', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart3 size={18} style={{ color: '#3b82f6' }} />
                <span>Feature Family Predictive Contribution (%)</span>
              </h3>

              <div style={{ width: '100%', height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={result.feature_families} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                    <XAxis dataKey="family" stroke="var(--text-secondary)" fontSize={12} />
                    <YAxis stroke="var(--text-secondary)" fontSize={12} unit="%" />
                    <Tooltip
                      contentStyle={{ background: 'var(--bg-surface-elevated)', border: '1px solid var(--border-color)', borderRadius: '6px' }}
                      labelStyle={{ color: 'var(--text-primary)', fontWeight: '600' }}
                    />
                    <Bar dataKey="contribution" radius={[4, 4, 0, 0]}>
                      {result.feature_families?.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={FEATURE_FAMILY_COLORS[entry.family] || '#3b82f6'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 4. FORENSIC FEATURE TABLE */}
            <div className="forensic-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: '600', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Layers size={18} style={{ color: '#3b82f6' }} />
                  <span>Attributed Visual Feature Evidence Table</span>
                </h3>

                {/* Search Input */}
                <div style={{ position: 'relative', width: '240px' }}>
                  <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                  <input
                    type="text"
                    placeholder="Search feature..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={{
                      width: '100%',
                      background: 'var(--bg-primary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '6px',
                      padding: '0.4rem 0.6rem 0.4rem 2rem',
                      fontSize: '0.8125rem',
                      color: 'var(--text-primary)'
                    }}
                  />
                </div>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '0.6rem' }}>Feature Name</th>
                      <th style={{ padding: '0.6rem' }}>Feature Family</th>
                      <th style={{ padding: '0.6rem' }}>SHAP Attribution</th>
                      <th style={{ padding: '0.6rem' }}>Directional Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getTableData().map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid var(--border-color-subtle)' }}>
                        <td className="font-mono" style={{ padding: '0.6rem', color: 'var(--text-primary)' }}>{row.feature}</td>
                        <td style={{ padding: '0.6rem' }}>
                          <span className="badge" style={{ background: 'rgba(59, 130, 246, 0.12)', color: '#3b82f6' }}>
                            {row.feature_family}
                          </span>
                        </td>
                        <td className="font-mono" style={{ padding: '0.6rem', fontWeight: '600', color: row.direction === 'FAKE' ? '#ef4444' : '#10b981' }}>
                          {row.shap_value > 0 ? `+${row.shap_value}` : row.shap_value}
                        </td>
                        <td style={{ padding: '0.6rem' }}>
                          <span className={row.direction === 'FAKE' ? 'badge badge-fake' : 'badge badge-real'}>
                            {row.direction === 'FAKE' ? 'Pushes FAKE' : 'Pushes REAL'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        )}
      </div>

      {/* MODEL INFO MODAL */}
      {showModelModal && (
        <div className="modal-overlay" onClick={() => setShowModelModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                Model & Benchmark Specification
              </h2>
              <button onClick={() => setShowModelModal(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
                <X size={20} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              <div>
                <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.4rem', fontWeight: '600' }}>Classifier Architecture</h4>
                <p>HistGradientBoostingClassifier trained on 216 handcrafted traditional visual features (Color HSV, LBP Micro-Texture, GLCM, Canny Edge, 2D DCT Frequency Energy).</p>
              </div>

              <div>
                <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.4rem', fontWeight: '600' }}>Leakage-Safe Partitioning</h4>
                <p>Splits are grouped at the <strong>Connected-Component Actor Graph Level</strong> to guarantee 0 subject/scene overlap across Train (70%), Validation (15%), and Test (15%) partitions.</p>
              </div>

              <div>
                <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.4rem', fontWeight: '600' }}>Held-Out Test Benchmark Metrics</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginTop: '0.5rem' }}>
                  <div style={{ background: 'var(--bg-primary)', padding: '0.6rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Balanced Accuracy</span>
                    <span className="font-mono" style={{ fontWeight: '600', color: '#f59e0b' }}>50.00%</span>
                  </div>
                  <div style={{ background: 'var(--bg-primary)', padding: '0.6rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>ROC-AUC</span>
                    <span className="font-mono" style={{ fontWeight: '600', color: '#f59e0b' }}>53.43%</span>
                  </div>
                  <div style={{ background: 'var(--bg-primary)', padding: '0.6rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>PR-AUC</span>
                    <span className="font-mono" style={{ fontWeight: '600', color: '#10b981' }}>85.17%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
