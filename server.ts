import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { db } from './src/dbMock';
import { GoogleGenAI, Type } from '@google/genai';
import nodemailer from 'nodemailer';

let aiInstance: GoogleGenAI | null = null;

function getGeminiClient() {
  if (!aiInstance) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn("WARNING: GEMINI_API_KEY is missing. Using local mathematical rule engine fallback.");
      return null;
    }
    aiInstance = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiInstance;
}

function runLocalRuleEngineFallback(data: {
  customerName: string;
  accountAgeDays: number;
  totalOrdersBefore: number;
  pastReturnsCount: number;
  itemCategory: 'Electronics' | 'Clothes' | 'Luxury' | 'Accessories';
  refundItemPrice: number;
}) {
  const riskScore = db.calculateRisk(
    data.accountAgeDays,
    data.totalOrdersBefore,
    data.pastReturnsCount,
    data.itemCategory,
    data.refundItemPrice
  );
  
  let finalAction: 'SAFE: AUTO-REFUND APPROVED' | 'WARNING: MANUAL CHECK REQUIRED' | 'FRAUD ALERT: REFUND BLOCKED' = 'WARNING: MANUAL CHECK REQUIRED';
  if (riskScore < 30) {
    finalAction = 'SAFE: AUTO-REFUND APPROVED';
  } else if (riskScore > 70) {
    finalAction = 'FRAUD ALERT: REFUND BLOCKED';
  }
  
  return {
    calculatedFraudRisk: riskScore,
    systemFinalAction: finalAction,
    explanation: "Assessed using local PRIJSM mathematical joint scoring GBDT rule engine fallback.",
    decisionFactors: [
      `Shopper Tenure Weight: ${data.accountAgeDays < 90 ? 'Elevated risk for brand new account' : 'Tenure security discount applied'}`,
      `Return History Weight: ${(data.totalOrdersBefore === 0 ? 100 : (data.pastReturnsCount / data.totalOrdersBefore * 100)).toFixed(0)}% return frequency logged`,
      `Product Category Weight: Volatility factor applied for ${data.itemCategory} segment`,
      `Price Weighting: Checked refund request value of ₹${data.refundItemPrice.toLocaleString()}`
    ]
  };
}

async function runGeminiRiskAnalysis(data: {
  customerName: string;
  accountAgeDays: number;
  totalOrdersBefore: number;
  pastReturnsCount: number;
  itemCategory: 'Electronics' | 'Clothes' | 'Luxury' | 'Accessories';
  refundItemPrice: number;
}) {
  const client = getGeminiClient();
  if (!client) {
    return runLocalRuleEngineFallback(data);
  }

  const prompt = `
    You are the PRIJSM (Predictive Refund Intelligence & Joint Scoring Model) AI Engine.
    Analyze the following refund request to assess fraud risk and return a secure score.
    
    Customer Profile:
    - Name: ${data.customerName}
    - Account Age: ${data.accountAgeDays} days
    - Total Purchase Orders Placed: ${data.totalOrdersBefore}
    - Past Refund Requests Logged: ${data.pastReturnsCount}
    
    Refund Request Details:
    - Product Category Segment: ${data.itemCategory}
    - Refund Item Price: ${data.refundItemPrice} INR
    
    Guidelines:
    1. New accounts (<90 days) with no or low order history requesting high refund prices are highly suspicious.
    2. High past returns relative to total orders represents severe return frequency warning.
    3. Luxury goods and electronics have high category volatility. Clothes are lower risk.
    4. Higher item value (especially > 20,000 INR) elevates the risk.
    5. Calculate a final riskScore between 0 and 100.
    6. Determine finalAction:
       - 'SAFE: AUTO-REFUND APPROVED' if riskScore is less than 30
       - 'WARNING: MANUAL CHECK REQUIRED' if riskScore is between 30 and 70 (inclusive)
       - 'FRAUD ALERT: REFUND BLOCKED' if riskScore is greater than 70
    7. Provide a plain, user-friendly, professional business English explanation. Do not use AI/ML jargon, summarize it so an auditor can read it.
  `;

  try {
    const apiCallPromise = client.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            riskScore: {
              type: Type.INTEGER,
              description: "The calculated refund fraud risk score between 0 and 100."
            },
            finalAction: {
              type: Type.STRING,
              description: "The gatekeeper's final action: 'SAFE: AUTO-REFUND APPROVED', 'WARNING: MANUAL CHECK REQUIRED', or 'FRAUD ALERT: REFUND BLOCKED'."
            },
            explanation: {
              type: Type.STRING,
              description: "A concise, user-friendly business explanation of the decision in plain English."
            },
            decisionFactors: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "A list of bullet points detailing the key risk factors (e.g., tenure, value, categories)."
            }
          },
          required: ["riskScore", "finalAction", "explanation", "decisionFactors"]
        }
      }
    });

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("Gemini API request exceeded 15000ms threshold.")), 15000);
    });

    const response = await Promise.race([apiCallPromise, timeoutPromise]);

    if (response.text) {
      const parsed = JSON.parse(response.text.trim());
      let finalAction = parsed.finalAction;
      if (!['SAFE: AUTO-REFUND APPROVED', 'WARNING: MANUAL CHECK REQUIRED', 'FRAUD ALERT: REFUND BLOCKED'].includes(finalAction)) {
        if (parsed.riskScore < 30) finalAction = 'SAFE: AUTO-REFUND APPROVED';
        else if (parsed.riskScore > 70) finalAction = 'FRAUD ALERT: REFUND BLOCKED';
        else finalAction = 'WARNING: MANUAL CHECK REQUIRED';
      }
      return {
        calculatedFraudRisk: parsed.riskScore,
        systemFinalAction: finalAction,
        explanation: parsed.explanation,
        decisionFactors: parsed.decisionFactors
      };
    }
  } catch (error: any) {
    console.error("Gemini API call failed or timed out, falling back to local math:", error?.message || error);
  }

  return runLocalRuleEngineFallback(data);
}

async function startServer() {
  // Sync the database with Firestore (or fallback to local disk) before serving requests
  try {
    await db.init();
  } catch (err) {
    console.error('Failed to initialize database connection:', err);
  }

  const app = express();
  const PORT = 3000;

  // Enable JSON body parsing
  app.use(express.json());

  // 1. API: Dashboard Metrics
  app.get('/api/dashboard/metrics', (req, res) => {
    try {
      const metrics = db.getMetrics();
      res.json(metrics);
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // 2. API: Get Requests List
  app.get('/api/requests', (req, res) => {
    try {
      const search = req.query.search as string || '';
      const category = req.query.category as string || '';
      const actionFilter = req.query.actionFilter as string || '';

      let list = db.getRequests();

      if (search) {
        const query = search.toLowerCase();
        list = list.filter(
          (r) =>
            r.customerName.toLowerCase().includes(query) ||
            r.id.toString() === query ||
            r.customerId.toString() === query
        );
      }

      if (category) {
        list = list.filter((r) => r.itemCategory === category);
      }

      if (actionFilter) {
        list = list.filter((r) => r.systemFinalAction === actionFilter);
      }

      res.json(list);
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // 3. API: Analyze Return (Draft/Simulation)
  app.post('/api/requests/analyze', async (req, res) => {
    try {
      const {
        customerName,
        accountAgeDays,
        totalOrdersBefore,
        pastReturnsCount,
        itemCategory,
        refundItemPrice,
      } = req.body;

      if (!customerName || typeof customerName !== 'string' || customerName.trim() === '') {
        return res.status(400).json({ error: 'Customer Name is required' });
      }

      const age = parseInt(accountAgeDays, 10);
      const orders = parseInt(totalOrdersBefore, 10);
      const returns = parseInt(pastReturnsCount, 10);
      const price = parseFloat(refundItemPrice);

      if (isNaN(age) || age < 0) {
        return res.status(400).json({ error: 'Valid Account Age in Days is required' });
      }
      if (isNaN(orders) || orders < 0) {
        return res.status(400).json({ error: 'Valid Total Orders count is required' });
      }
      if (isNaN(returns) || returns < 0) {
        return res.status(400).json({ error: 'Valid Past Returns count is required' });
      }
      if (returns > orders) {
        return res.status(400).json({ error: 'Past Returns count cannot exceed Total Orders Placed' });
      }
      if (isNaN(price) || price <= 0) {
        return res.status(400).json({ error: 'Valid Refund Item Price (INR) is required' });
      }
      if (!['Electronics', 'Clothes', 'Luxury', 'Accessories'].includes(itemCategory)) {
        return res.status(400).json({ error: 'Item Category must be Electronics, Clothes, Luxury, or Accessories' });
      }

      // Run AI/ML Gemini analysis
      const analysis = await runGeminiRiskAnalysis({
        customerName,
        accountAgeDays: age,
        totalOrdersBefore: orders,
        pastReturnsCount: returns,
        itemCategory: itemCategory as 'Electronics' | 'Clothes' | 'Luxury' | 'Accessories',
        refundItemPrice: price,
      });

      res.json({
        id: -1, // Indicates a transient draft
        customerId: -1,
        customerName,
        itemCategory,
        refundItemPrice: price,
        calculatedFraudRisk: analysis.calculatedFraudRisk,
        systemFinalAction: analysis.systemFinalAction,
        createdAt: new Date().toISOString(),
        explanation: analysis.explanation,
        decisionFactors: analysis.decisionFactors,
        formParams: {
          customerName,
          accountAgeDays: age,
          totalOrdersBefore: orders,
          pastReturnsCount: returns,
          itemCategory,
          refundItemPrice: price
        }
      });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // 3b. API: Save Analyzed Request to Database Ledger (Explicit Commit)
  app.post('/api/requests/save', (req, res) => {
    try {
      const {
        customerName,
        accountAgeDays,
        totalOrdersBefore,
        pastReturnsCount,
        itemCategory,
        refundItemPrice,
        calculatedFraudRisk,
        systemFinalAction,
      } = req.body;

      if (!customerName || typeof customerName !== 'string' || customerName.trim() === '') {
        return res.status(400).json({ error: 'Customer Name is required' });
      }

      const age = parseInt(accountAgeDays, 10);
      const orders = parseInt(totalOrdersBefore, 10);
      const returns = parseInt(pastReturnsCount, 10);
      const price = parseFloat(refundItemPrice);
      const score = parseInt(calculatedFraudRisk, 10);

      if (isNaN(age) || age < 0 || isNaN(orders) || isNaN(returns) || isNaN(price) || isNaN(score)) {
        return res.status(400).json({ error: 'Invalid or incomplete profile details' });
      }

      const savedRequest = db.saveRequestToLedger({
        customerName,
        accountAgeDays: age,
        totalOrdersBefore: orders,
        pastReturnsCount: returns,
        itemCategory: itemCategory as 'Electronics' | 'Clothes' | 'Luxury' | 'Accessories',
        refundItemPrice: price,
        calculatedFraudRisk: score,
        systemFinalAction,
      });

      res.status(201).json(savedRequest);
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // 3c. API: Sync Local Storage Data on startup
  app.post('/api/requests/sync-local', (req, res) => {
    try {
      const { clientRequests } = req.body;
      if (Array.isArray(clientRequests)) {
        db.syncLocalData(clientRequests);
      }
      res.json({ success: true });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // 4. API: Business Report Chart Data
  app.get('/api/reports/business', (req, res) => {
    try {
      const reports = db.getReports();
      res.json(reports);
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // 5. API: Download Filtered CSV String
  app.get('/api/requests/csv', (req, res) => {
    try {
      const search = req.query.search as string || '';
      const category = req.query.category as string || '';
      const actionFilter = req.query.actionFilter as string || '';

      let list = db.getRequests();

      if (search) {
        const query = search.toLowerCase();
        list = list.filter(
          (r) =>
            r.customerName.toLowerCase().includes(query) ||
            r.id.toString() === query ||
            r.customerId.toString() === query
        );
      }

      if (category) {
        list = list.filter((r) => r.itemCategory === category);
      }

      if (actionFilter) {
        list = list.filter((r) => r.systemFinalAction === actionFilter);
      }

      const csv = db.getCsvData(list);
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', 'attachment; filename=prijsm_audit_ledger.csv');
      res.send(csv);
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // 6. API: Send Filtered CSV Report via Email
  app.post('/api/requests/email', async (req, res) => {
    try {
      const { email, search, category, actionFilter } = req.body;
      if (!email || typeof email !== 'string' || !email.includes('@')) {
        return res.status(400).json({ error: 'A valid email address is required.' });
      }

      let list = db.getRequests();

      if (search) {
        const query = search.toLowerCase();
        list = list.filter(
          (r) =>
            r.customerName.toLowerCase().includes(query) ||
            r.id.toString() === query ||
            r.customerId.toString() === query
        );
      }

      if (category) {
        list = list.filter((r) => r.itemCategory === category);
      }

      if (actionFilter) {
        list = list.filter((r) => r.systemFinalAction === actionFilter);
      }

      const csv = db.getCsvData(list);

      const smtpHost = process.env.SMTP_HOST;
      const smtpPort = process.env.SMTP_PORT ? parseInt(process.env.SMTP_PORT) : 465;
      const smtpUser = process.env.SMTP_USER;
      const smtpPass = process.env.SMTP_PASS;

      let emailSent = false;
      let isSimulated = true;

      if (smtpHost && smtpUser && smtpPass && !smtpUser.includes('your-gmail-address')) {
        try {
          const transporter = nodemailer.createTransport({
            host: smtpHost,
            port: smtpPort,
            secure: smtpPort === 465,
            auth: {
              user: smtpUser,
              pass: smtpPass,
            },
            timeout: 8000 // 8 second timeout
          } as any);

          await transporter.sendMail({
            from: `"PRIJSM Security Monitor" <${smtpUser}>`,
            to: email,
            subject: `📊 PRIJSM Return Security Audit Report - ${new Date().toLocaleDateString()}`,
            text: `Hello,

Attached is your Return Security and Fraud Mitigation Audit Report dispatched from your PRIJSM local instance.

- Total Checked Rows: ${list.length}
- Target Email Address: ${email}
- Security Environment: Sandbox Active Ledger

If you have any questions or require further risk classification diagnostics, please consult your main PRIJSM Command Center.

Best regards,
PRIJSM Security Agent`,
            attachments: [
              {
                filename: 'prijsm_audit_ledger.csv',
                content: csv,
              }
            ]
          });
          isSimulated = false;
          emailSent = true;
          console.log(`[PRIJSM MAILER] Real email dispatched successfully via SMTP to ${email}`);
        } catch (mailError: any) {
          console.error(`[PRIJSM MAILER] Real SMTP dispatch failed:`, mailError.message);
          return res.status(500).json({
            error: `Failed to dispatch real email: ${mailError.message}. Make sure your App Password is correct in the Secrets panel.`
          });
        }
      } else {
        // Fallback simulation mode
        isSimulated = true;
        emailSent = true;
      }

      // Print secure enterprise dispatch diagnostics to stdout
      console.log('===================================================');
      console.log(`[PRIJSM MAILER] Preparing to dispatch audit spreadsheet...`);
      console.log(`Recipient: ${email}`);
      console.log(`Report Rows: ${list.length}`);
      console.log(`Spreadsheet Size: ${(Buffer.byteLength(csv) / 1024).toFixed(2)} KB`);
      console.log(`Attachment Name: prijsm_audit_ledger.csv`);
      console.log(`SMTP Mode: ${isSimulated ? 'SIMULATED (Sandbox Fallback)' : 'REAL DISPATCH (Active Connection)'}`);
      console.log(`Status: SMTP Dispatched Successfully!`);
      console.log('===================================================');

      res.json({
        success: true,
        isSimulated: isSimulated,
        message: isSimulated
          ? `The spreadsheet audit report (${list.length} rows) was compiled successfully. Because your real SMTP credentials (SMTP_USER, SMTP_PASS, etc.) are not yet configured in the AI Studio Secrets panel, the dispatch was SIMULATED. Please add your credentials in Settings/Secrets to receive actual CSV files in your inbox!`
          : `The spreadsheet audit report (${list.length} rows) was compiled successfully and sent directly to ${email} via your active SMTP credentials!`
      });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // Serve static files / Vite dev middleware
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[PRIJSM] Server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start PRIJSM Server:', err);
});
