import express from 'express';
import pkg from 'mongodb';
const { MongoClient } = pkg;
import OpenAI from 'openai';
import cors from 'cors';
import * as dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "@langchain/community/vectorstores/mongodb_atlas";
import { OpenAIEmbeddings } from "@langchain/openai";
import fetch from 'node-fetch';
import { encoding_for_model } from '@dqbd/tiktoken';

dotenv.config();


globalThis.fetch = fetch;

const app = express();
const PORT = process.env.PORT || 3000;

// Token encoder for cost estimation
const enc = encoding_for_model('gpt-3.5-turbo');

// OpenAI pricing (per 1K tokens) - updated as of 2024
const PRICING = {
  'gpt-3.5-turbo': {
    input: 0.0005,
    output: 0.0015
  },
  'text-embedding-3-large': {
    input: 0.00013,
    output: 0
  }
};

// Utility functions for cost calculation
function countTokens(text) {
  try {
    return enc.encode(text).length;
  } catch (error) {
    // Fallback estimation: ~4 chars per token
    return Math.ceil(text.length / 4);
  }
}

function calculateCost(inputTokens, outputTokens, model) {
  const pricing = PRICING[model] || PRICING['gpt-3.5-turbo'];
  const inputCost = (inputTokens / 1000) * pricing.input;
  const outputCost = (outputTokens / 1000) * pricing.output;
  return {
    inputCost,
    outputCost,
    totalCost: inputCost + outputCost,
    inputTokens,
    outputTokens,
    totalTokens: inputTokens + outputTokens
  };
}

// Request timing middleware
app.use((req, res, next) => {
  req.startTime = Date.now();
  const originalSend = res.send;
  
  res.send = function(data) {
    const duration = Date.now() - req.startTime;
    
    // Add timing to response if it's JSON
    if (res.getHeader('content-type')?.includes('application/json')) {
      try {
        const jsonData = typeof data === 'string' ? JSON.parse(data) : data;
        if (typeof jsonData === 'object' && jsonData !== null) {
          jsonData.timing = {
            requestDuration: duration,
            timestamp: new Date().toISOString()
          };
          data = JSON.stringify(jsonData);
        }
      } catch (e) {
        // If parsing fails, just add timing info to console
        console.log(`Request to ${req.path} took ${duration}ms`);
      }
    }
    
    originalSend.call(this, data);
  };
  
  next();
});

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = 'uploads/';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {

    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
   
    const allowedTypes = ['.txt', '.pdf', '.doc', '.docx'];
    const fileExt = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(fileExt)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only .txt, .pdf, .doc, .docx files are allowed.'));
    }
  },
  limits: {
    fileSize: 10 * 1024 * 1024 //10mb limit
  }
});


app.use(cors());
app.use(express.json());


const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});


const client = new MongoClient(process.env.MONGODB_URI);
let db, collection, chatHistoryCollection, documentsCollection;


async function connectDB() {
  try {
    await client.connect();
    db = client.db("TalkingLibrary");
    collection = db.collection("Baalkaand");
    chatHistoryCollection = db.collection("ChatHistory");
    documentsCollection = db.collection("Documents");
    console.log("Connected to MongoDB");
  } catch (error) {
    console.error("MongoDB connection error:", error);
    process.exit(1);
  }
}

async function generateEmbedding(text) {
  try {
    const startTime = Date.now();
    const inputTokens = countTokens(text);
    
    const response = await openai.embeddings.create({
      model: "text-embedding-3-large",
      input: text,
    });
    
    const embeddingTime = Date.now() - startTime;
    const cost = calculateCost(inputTokens, 0, 'text-embedding-3-large');
    
    console.log(`Embedding generated - Tokens: ${inputTokens}, Cost: $${cost.totalCost.toFixed(6)}, Time: ${embeddingTime}ms`);
    
    return {
      embedding: response.data[0].embedding,
      metrics: {
        tokens: inputTokens,
        cost: cost.totalCost,
        duration: embeddingTime
      }
    };
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw error;
  }
}


async function generateEmbeddingsFromSource(source, collection, documentId) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 8192,
    chunkOverlap: 200,
  });

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "text-embedding-3-large",
  });

  let documents;

  try {
    if (source.type === "file") {
      const loader = new TextLoader(source.path);
      const docs = await loader.load();
      documents = await splitter.splitDocuments(docs);
    } else if (source.type === "pdf") {
      const loader = new PDFLoader(source.path);
      const docs = await loader.load();
      documents = await splitter.splitDocuments(docs);
    } else if (source.type === "url") {
      const loader = new CheerioWebBaseLoader(source.url);
      const docs = await loader.load();
      documents = await splitter.splitDocuments(docs);
    } else {
      throw new Error("Invalid source type. Use 'file', 'pdf', or 'url'");
    }

    documents = documents.map(doc => ({
      ...doc,
      metadata: {
        ...doc.metadata,
        documentId: documentId,
        source: source.url || source.path,
        filename: source.filename || 'unknown',
        type: source.type
      }
    }));

    const vectorStore = await MongoDBAtlasVectorSearch.fromDocuments(
      documents,
      embeddings,
      { collection }
    );

    console.log(`Successfully processed ${source.type}: ${source.path || source.url}`);
    return {
      success: true,
      documentsProcessed: documents.length,
      vectorStore
    };
  } catch (error) {
    console.error(`Error processing ${source.type}: ${source.path || source.url}`, error);
    throw error;
  }
}

async function vectorSearch(query, limit = 5) {
  try {
    const searchStartTime = Date.now();
    const embeddingResult = await generateEmbedding(query);
    const queryEmbedding = embeddingResult.embedding;
    
    const pipeline = [
      {
        $vectorSearch: {
          index: "default",
          path: "embedding",
          queryVector: queryEmbedding,
          numCandidates: 100,
          limit: limit
        }
      },
      {
        $project: {
          text: 1,
          score: { $meta: "vectorSearchScore" },
          _id: 0
        }
      }
    ];

    const results = await collection.aggregate(pipeline).toArray();
    const searchDuration = Date.now() - searchStartTime;
    
    return {
      results,
      metrics: {
        embeddingMetrics: embeddingResult.metrics,
        searchDuration,
        totalDuration: searchDuration,
        resultsCount: results.length
      }
    };
  } catch (error) {
    console.error("Vector search error:", error);
    throw error;
  }
}


async function getChatHistory(sessionId) {
  try {
    const messages = await chatHistoryCollection
      .find({ sessionId })
      .sort({ timestamp: 1 })
      .limit(10)
      .toArray();
    
    return messages.map(msg => ({
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp
    }));
  } catch (error) {
    console.error("Error fetching chat history:", error);
    return [];
  }
}

async function addToHistory(sessionId, role, content) {
  try {
    const message = {
      sessionId,
      role,
      content,
      timestamp: new Date()
    };
    
    await chatHistoryCollection.insertOne(message);
    
    
    const messageCount = await chatHistoryCollection.countDocuments({ sessionId });
    if (messageCount > 10) {
      const oldMessages = await chatHistoryCollection
        .find({ sessionId })
        .sort({ timestamp: 1 })
        .limit(messageCount - 10)
        .toArray();
      
      const idsToDelete = oldMessages.map(msg => msg._id);
      await chatHistoryCollection.deleteMany({ _id: { $in: idsToDelete } });
    }
  } catch (error) {
    console.error("Error adding to chat history:", error);
  }
}


async function rerankResults(query, searchResults, limit = 3) {
  try {
    if (searchResults.length <= 1) {
      return {
        results: searchResults,
        metrics: {
          tokens: 0,
          cost: 0,
          duration: 0
        }
      };
    }

    const rerankStartTime = Date.now();
    const documentsText = searchResults.map((result, index) => 
      `Document ${index + 1}:\n${result.text.substring(0, 800)}...`
    ).join('\n\n---\n\n');

    const rerankPrompt = `Given the user query and the following documents, rank them from most relevant (1) to least relevant based on how well they answer the query.

User Query: "${query}"

Documents:
${documentsText}

Respond with ONLY a JSON array of document numbers ordered by relevance (most relevant first).
Example: [2, 1, 3]`;

    const systemMessage = "You are a document relevance ranking expert. Analyze the query and documents to provide accurate relevance rankings.";
    const inputTokens = countTokens(systemMessage + rerankPrompt);
    
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: systemMessage },
        { role: "user", content: rerankPrompt }
      ],
      temperature: 0.1,
      max_tokens: 100
    });

    const outputTokens = countTokens(response.choices[0].message.content);
    const cost = calculateCost(inputTokens, outputTokens, 'gpt-3.5-turbo');
    const rerankDuration = Date.now() - rerankStartTime;
    
    let ranking;
    try {
      ranking = JSON.parse(response.choices[0].message.content.trim());
    } catch (parseError) {
      console.error("Failed to parse reranking response:", response.choices[0].message.content);
      return {
        results: searchResults,
        metrics: {
          tokens: cost.totalTokens,
          cost: cost.totalCost,
          duration: rerankDuration,
          error: "Failed to parse reranking response"
        }
      };
    }

    const rerankedResults = [];
    for (const docNum of ranking) {
      const index = docNum - 1; 
      if (index >= 0 && index < searchResults.length) {
        rerankedResults.push({
          ...searchResults[index],
          originalScore: searchResults[index].score,
          rerankPosition: rerankedResults.length + 1
        });
      }
    }

    searchResults.forEach((result, index) => {
      if (!rerankedResults.find(r => r.text === result.text)) {
        rerankedResults.push({
          ...result,
          originalScore: result.score,
          rerankPosition: rerankedResults.length + 1
        });
      }
    });

    console.log(`Reranking completed - Tokens: ${cost.totalTokens}, Cost: $${cost.totalCost.toFixed(6)}, Time: ${rerankDuration}ms`);
    
    return {
      results: rerankedResults.slice(0, limit),
      metrics: {
        tokens: cost.totalTokens,
        cost: cost.totalCost,
        duration: rerankDuration
      }
    };
  } catch (error) {
    console.error("Error during reranking:", error);
    return {
      results: searchResults,
      metrics: {
        tokens: 0,
        cost: 0,
        duration: 0,
        error: error.message
      }
    };
  }
}


async function generateResponseWithCitations(query, searchResults, chatHistory) {
  try {
    const responseStartTime = Date.now();
    
    const contextWithCitations = searchResults
      .map((result, index) => {
        const citationNumber = index + 1;
        const scoreInfo = result.rerankPosition 
          ? `Rank: ${result.rerankPosition}, Original Score: ${result.originalScore.toFixed(3)}`
          : `Score: ${result.score.toFixed(3)}`;
        return `[${citationNumber}] ${result.text}\n(Source ${citationNumber}: ${scoreInfo})`;
      })
      .join('\n\n');

    const systemPrompt = `You are Oliver, a document assistant. You have access to numbered document sources and should answer questions based on the provided context.

IMPORTANT RULES:
1. Always answer based on the provided context sources
2. Use inline citations [1], [2], [3] etc. after statements to reference the source numbers
3. Every factual claim should have a citation from the sources provided
4. If the context doesn't contain information about the query, say "The document doesn't appear to have information about that."
5. Maintain a polite, knowledgeable persona
6. Keep responses concise but informative
7. Only cite sources that are actually provided in the context

Available Sources:
${contextWithCitations}

Example of proper citation format:
- "The company was founded in 1995 [1] and has grown significantly over the years [2]."
- "According to the documentation [1], the process involves three main steps [3]."`;

    const messages = [
      { role: "system", content: systemPrompt },
      ...chatHistory.slice(-6), 
      { role: "user", content: query }
    ];

    // Calculate input tokens
    const inputText = messages.map(m => m.content).join('\n');
    const inputTokens = countTokens(inputText);
    
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: messages,
      temperature: 0.3,
      max_tokens: 400
    });

    const responseText = response.choices[0].message.content;
    const outputTokens = countTokens(responseText);
    const cost = calculateCost(inputTokens, outputTokens, 'gpt-3.5-turbo');
    const responseDuration = Date.now() - responseStartTime;
    
   
    const citationRegex = /\[(\d+)\]/g;
    const foundCitations = new Set();
    let match;
    
    while ((match = citationRegex.exec(responseText)) !== null) {
      const citationNum = parseInt(match[1]);
      if (citationNum > 0 && citationNum <= searchResults.length) {
        foundCitations.add(citationNum);
      }
    }

    
    const sourceSnippets = Array.from(foundCitations)
      .sort((a, b) => a - b)
      .map(citationNum => {
        const result = searchResults[citationNum - 1];
        return {
          citationNumber: citationNum,
          text: result.text,
          originalScore: result.originalScore || result.score,
          rerankPosition: result.rerankPosition,
          isReranked: !!result.rerankPosition
        };
      });

    console.log(`Generated response - Citations: ${foundCitations.size}, Tokens: ${cost.totalTokens}, Cost: $${cost.totalCost.toFixed(6)}, Time: ${responseDuration}ms`);

    return {
      response: responseText,
      citations: Array.from(foundCitations).sort(),
      sourceSnippets: sourceSnippets,
      metrics: {
        tokens: cost.totalTokens,
        cost: cost.totalCost,
        duration: responseDuration
      }
    };
  } catch (error) {
    console.error("Error generating response with citations:", error);
    throw error;
  }
}


async function generateResponse(query, context, chatHistory) {
  try {
    const systemPrompt = `You are Oliver a dosument assitant. You have access to the document and should answer questions based on the provided context.

IMPORTANT RULES:
1. Always answer based on the provided context
2. If the context doesn't contain information about the query, say "The document posssibly doesn't have any information about that."
3. Maintain the persona of a polite assistant - be respectful, devotional, and knowledgeable
5. Keep responses concise but informative

Context:
${context}`;

    const messages = [
      { role: "system", content: systemPrompt },
      ...chatHistory.slice(-6), 
      { role: "user", content: query }
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: messages,
      temperature: 0.3,
      max_tokens: 300
    });

    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error generating response:", error);
    throw error;
  }
}



// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: ' Chatbot Server is running' });
});


app.post('/upload', upload.single('document'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const { url } = req.body;
    const filePath = req.file.path;
    const originalName = req.file.originalname;
    const fileExtension = path.extname(originalName).toLowerCase();

    console.log(`Processing uploaded file: ${originalName}`);

    
    const documentId = `doc_${Date.now()}_${Math.round(Math.random() * 1E9)}`;

    let source;

    
    if (url && url.trim()) {
      source = {
        type: "url",
        url: url.trim(),
        filename: originalName
      };
    } else {
      
      if (fileExtension === '.pdf') {
        source = {
          type: "pdf",
          path: filePath,
          filename: originalName
        };
      } else if (['.txt', '.doc', '.docx'].includes(fileExtension)) {
        source = {
          type: "file",
          path: filePath,
          filename: originalName
        };
      } else {
        
        fs.unlinkSync(filePath);
        return res.status(400).json({ error: 'Unsupported file type' });
      }
    }

    
    const result = await generateEmbeddingsFromSource(source, collection, documentId);

    
    const documentMetadata = {
      filename: originalName,
      type: source.type,
      source: source.url || originalName,
      documentsProcessed: result.documentsProcessed,
      uploadedAt: new Date(),
      documentId: documentId
    };

    await documentsCollection.insertOne(documentMetadata);

    
    if (source.type !== "url" && fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }

    res.json({
      success: true,
      message: `Document processed successfully`,
      filename: originalName,
      documentsProcessed: result.documentsProcessed,
      type: source.type,
      source: source.url || originalName,
      documentId: documentMetadata.documentId
    });

  } catch (error) {
    console.error("Upload processing error:", error);
    
    
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({ 
      error: 'Failed to process document',
      details: error.message 
    });
  }
});

app.post('/chat', async (req, res) => {
  try {
    const { message, sessionId = 'default' } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    const chatStartTime = Date.now();
    const chatHistory = await getChatHistory(sessionId);

    // Perform vector search with metrics
    const searchData = await vectorSearch(message, 10);
    const searchResults = searchData.results || searchData; // Handle both old and new format
    
    if (searchResults.length === 0) {
      const response = "The document posssibly doesn't have any information about that.";
      await addToHistory(sessionId, 'user', message);
      await addToHistory(sessionId, 'assistant', response);
      
      const totalDuration = Date.now() - chatStartTime;
      return res.json({
        response,
        sessionId,
        searchResults: [],
        metrics: {
          totalDuration,
          searchMetrics: searchData.metrics || null,
          totalCost: searchData.metrics?.cost || 0,
          totalTokens: searchData.metrics?.tokens || 0
        }
      });
    }

    // Rerank results with metrics
    const rerankData = await rerankResults(message, searchResults, 3);
    const rerankedResults = rerankData.results || rerankData; // Handle both old and new format

    // Generate response with citations and metrics
    const responseWithCitations = await generateResponseWithCitations(message, rerankedResults, chatHistory);

    // Save to history
    await addToHistory(sessionId, 'user', message);
    await addToHistory(sessionId, 'assistant', responseWithCitations.response);

    const totalDuration = Date.now() - chatStartTime;
    const totalCost = (searchData.metrics?.cost || 0) + (rerankData.metrics?.cost || 0) + (responseWithCitations.metrics?.cost || 0);
    const totalTokens = (searchData.metrics?.tokens || 0) + (rerankData.metrics?.tokens || 0) + (responseWithCitations.metrics?.tokens || 0);

    res.json({
      response: responseWithCitations.response,
      citations: responseWithCitations.citations,
      sourceSnippets: responseWithCitations.sourceSnippets,
      sessionId,
      searchResults: rerankedResults.map(r => ({
        text: r.text.substring(0, 200) + '...', 
        originalScore: r.originalScore || r.score,
        rerankPosition: r.rerankPosition,
        isReranked: !!r.rerankPosition
      })),
      metrics: {
        totalDuration,
        totalCost,
        totalTokens,
        breakdown: {
          search: searchData.metrics || null,
          rerank: rerankData.metrics || null,
          response: responseWithCitations.metrics || null
        }
      }
    });

  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


app.get('/chat/history/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const history = await getChatHistory(sessionId);
    
    res.json({
      sessionId,
      history: history.map(msg => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp
      }))
    });
  } catch (error) {
    console.error("History fetch error:", error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


app.delete('/chat/history/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    await chatHistoryCollection.deleteMany({ sessionId });
    
    res.json({
      message: `Chat history cleared for session: ${sessionId}`
    });
  } catch (error) {
    console.error("History clear error:", error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


app.get('/sessions', async (req, res) => {
  try {
    const sessionsData = await chatHistoryCollection.aggregate([
      {
        $group: {
          _id: "$sessionId",
          messageCount: { $sum: 1 },
          lastActivity: { $max: "$timestamp" }
        }
      },
      {
        $project: {
          sessionId: "$_id",
          messageCount: 1,
          lastActivity: 1,
          _id: 0
        }
      },
      {
        $sort: { lastActivity: -1 }
      }
    ]).toArray();

    res.json({ sessions: sessionsData });
  } catch (error) {
    console.error("Sessions fetch error:", error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


app.post('/search', async (req, res) => {
  try {
    const { query, limit = 3, useReranking = false } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    const searchStartTime = Date.now();
    const searchLimit = useReranking ? Math.max(limit * 3, 10) : limit;
    const searchData = await vectorSearch(query, searchLimit);
    const results = searchData.results || searchData; // Handle both formats
    
    let finalResults = results;
    let rerankMetrics = null;
    
    if (useReranking && results.length > 1) {
      const rerankData = await rerankResults(query, results, limit);
      finalResults = rerankData.results || rerankData;
      rerankMetrics = rerankData.metrics;
    }
    
    const totalDuration = Date.now() - searchStartTime;
    const totalCost = (searchData.metrics?.cost || 0) + (rerankMetrics?.cost || 0);
    const totalTokens = (searchData.metrics?.tokens || 0) + (rerankMetrics?.tokens || 0);
    
    res.json({
      query,
      useReranking,
      totalCandidates: results.length,
      results: finalResults.map(r => ({
        text: r.text,
        originalScore: r.originalScore || r.score,
        rerankPosition: r.rerankPosition,
        isReranked: !!r.rerankPosition
      })),
      metrics: {
        totalDuration,
        totalCost,
        totalTokens,
        breakdown: {
          search: searchData.metrics || null,
          rerank: rerankMetrics
        }
      }
    });
  } catch (error) {
    console.error("Search error:", error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


app.post('/test-citations', async (req, res) => {
  try {
    const { query, limit = 3 } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    const testStartTime = Date.now();
    const searchData = await vectorSearch(query, Math.max(limit * 3, 10));
    const searchResults = searchData.results || searchData;
    
    if (searchResults.length === 0) {
      return res.json({
        query,
        message: "No documents found",
        citations: [],
        sourceSnippets: [],
        metrics: {
          totalDuration: Date.now() - testStartTime,
          totalCost: searchData.metrics?.cost || 0,
          totalTokens: searchData.metrics?.tokens || 0
        }
      });
    }

    const rerankData = await rerankResults(query, searchResults, limit);
    const rerankedResults = rerankData.results || rerankData;
    const responseWithCitations = await generateResponseWithCitations(query, rerankedResults, []);

    const totalDuration = Date.now() - testStartTime;
    const totalCost = (searchData.metrics?.cost || 0) + (rerankData.metrics?.cost || 0) + (responseWithCitations.metrics?.cost || 0);
    const totalTokens = (searchData.metrics?.tokens || 0) + (rerankData.metrics?.tokens || 0) + (responseWithCitations.metrics?.tokens || 0);

    res.json({
      query,
      response: responseWithCitations.response,
      citations: responseWithCitations.citations,
      sourceSnippets: responseWithCitations.sourceSnippets,
      totalCandidates: searchResults.length,
      finalResults: limit,
      metrics: {
        totalDuration,
        totalCost,
        totalTokens,
        breakdown: {
          search: searchData.metrics || null,
          rerank: rerankData.metrics || null,
          response: responseWithCitations.metrics || null
        }
      }
    });
  } catch (error) {
    console.error("Test citations error:", error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


app.get('/documents', async (req, res) => {
  try {
    const documents = await documentsCollection
      .find({})
      .sort({ uploadedAt: -1 })
      .toArray();
    
    res.json({
      documents: documents.map(doc => ({
        documentId: doc.documentId,
        filename: doc.filename,
        type: doc.type,
        source: doc.source,
        documentsProcessed: doc.documentsProcessed,
        uploadedAt: doc.uploadedAt
      }))
    });
  } catch (error) {
    console.error("Documents fetch error:", error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


app.get('/debug/embeddings/:documentId', async (req, res) => {
  try {
    const { documentId } = req.params;
    
    
    const sampleEmbeddings = await collection.find({
      $or: [
        { "metadata.documentId": documentId },
        { "documentId": documentId },
        { "metadata.document_id": documentId },
        { "document_id": documentId }
      ]
    }).limit(3).toArray();
    
    
    const randomEmbeddings = await collection.find({}).limit(2).toArray();
    
    res.json({
      documentId,
      sampleRelatedEmbeddings: sampleEmbeddings.map(doc => ({
        _id: doc._id,
        metadata: doc.metadata,
        text: doc.text ? doc.text.substring(0, 100) + '...' : 'N/A',
        hasEmbedding: !!doc.embedding
      })),
      sampleRandomEmbeddings: randomEmbeddings.map(doc => ({
        _id: doc._id,
        metadata: doc.metadata,
        text: doc.text ? doc.text.substring(0, 100) + '...' : 'N/A',
        hasEmbedding: !!doc.embedding
      })),
      totalEmbeddingsInCollection: await collection.countDocuments({})
    });
  } catch (error) {
    console.error("Debug endpoint error:", error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});


app.delete('/documents/:documentId', async (req, res) => {
  try {
    const { documentId } = req.params;
    
    
    const document = await documentsCollection.findOne({ documentId });
    if (!document) {
      return res.status(404).json({ error: 'Document not found' });
    }

    
    console.log(`Attempting to delete embeddings for documentId: ${documentId}`);
    
    
    const possibleFilters = [
      { "metadata.documentId": documentId },
      { "documentId": documentId },
      { "metadata.document_id": documentId },
      { "document_id": documentId }
    ];
    
    let embeddingDeleteResult = { deletedCount: 0 };
    let filterUsed = null;
    
    
    for (const filter of possibleFilters) {
      const testResult = await collection.countDocuments(filter);
      console.log(`Filter ${JSON.stringify(filter)} matches ${testResult} documents`);
      
      if (testResult > 0) {
        embeddingDeleteResult = await collection.deleteMany(filter);
        filterUsed = filter;
        console.log(`Deleted ${embeddingDeleteResult.deletedCount} embeddings using filter: ${JSON.stringify(filter)}`);
        break;
      }
    }
    
    
    if (embeddingDeleteResult.deletedCount === 0) {
      console.log("No embeddings found with standard filters. Searching for any documents with this documentId in any field...");
      
      
      const comprehensiveFilter = {
        $or: [
          { "metadata.documentId": documentId },
          { "documentId": documentId },
          { "metadata.document_id": documentId },
          { "document_id": documentId },
          { "metadata": { $exists: true, $regex: documentId } },
          { $text: { $search: documentId } }
        ]
      };
      
      try {
        const comprehensiveResult = await collection.countDocuments(comprehensiveFilter);
        console.log(`Comprehensive search found ${comprehensiveResult} documents`);
        
        if (comprehensiveResult > 0) {
          embeddingDeleteResult = await collection.deleteMany(comprehensiveFilter);
          filterUsed = comprehensiveFilter;
          console.log(`Deleted ${embeddingDeleteResult.deletedCount} embeddings using comprehensive filter`);
        }
      } catch (searchError) {
        console.log("Comprehensive search failed, continuing with standard deletion");
      }
    }
    
    
    await documentsCollection.deleteOne({ documentId });
    
    const responseMessage = embeddingDeleteResult.deletedCount > 0 
      ? `Document deleted successfully` 
      : `Document metadata deleted, but no embeddings were found to delete (they may have been already cleaned up)`;
    
    res.json({
      success: true,
      message: responseMessage,
      filename: document.filename,
      embeddingsDeleted: embeddingDeleteResult.deletedCount,
      filterUsed: filterUsed ? JSON.stringify(filterUsed) : "none"
    });
  } catch (error) {
    console.error("Document deletion error:", error);
    res.status(500).json({ 
      error: 'Internal server error',
      details: error.message 
    });
  }
});


async function startServer() {
  await connectDB();
  
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
    console.log(`Chat endpoint: POST http://localhost:${PORT}/chat`);
    console.log(`Search endpoint: POST http://localhost:${PORT}/search`);
    console.log(`Upload endpoint: POST http://localhost:${PORT}/upload`);
  });
}


process.on('SIGINT', async () => {
  console.log('Shutting down gracefully...');
  await client.close();
  process.exit(0);
});

startServer().catch(console.error);