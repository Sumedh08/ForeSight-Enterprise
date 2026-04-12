/**
 * Dynamic Cube.js Schema Generator
 * 
 * This file auto-generates Cube schemas from ALL tables in PostgreSQL.
 * No hardcoding — any table uploaded to Postgres gets an auto-generated Cube model.
 */

const { Pool } = require("pg");

const pool = new Pool({
  host: process.env.CUBEJS_DB_HOST || "postgres",
  port: parseInt(process.env.CUBEJS_DB_PORT || "5432"),
  database: process.env.CUBEJS_DB_NAME || "natwest_db",
  user: process.env.CUBEJS_DB_USER || "admin",
  password: process.env.CUBEJS_DB_PASS || "adminpassword",
});

asyncModule(async () => {
  // 1. Discover all user tables from Postgres information_schema
  const { rows: tables } = await pool.query(`
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_type = 'BASE TABLE'
    ORDER BY table_name
  `);

  for (const { table_name } of tables) {
    // 2. Get columns for each table
    const { rows: columns } = await pool.query(`
      SELECT column_name, data_type, is_nullable
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = $1
      ORDER BY ordinal_position
    `, [table_name]);

    // 3. Classify columns into measures and dimensions
    const numericTypes = ["integer", "bigint", "smallint", "numeric", "real", "double precision"];
    const dateTypes = ["timestamp without time zone", "timestamp with time zone", "date"];
    const textTypes = ["character varying", "text", "character", "boolean"];

    const measures = {};
    const dimensions = {};

    // Always add a row count measure
    measures.count = { type: "count" };

    for (const col of columns) {
      const safeName = col.column_name.replace(/[^a-zA-Z0-9_]/g, "_");

      if (dateTypes.includes(col.data_type)) {
        dimensions[safeName] = {
          sql: `"${col.column_name}"`,
          type: "time",
          title: col.column_name,
        };
      } else if (numericTypes.includes(col.data_type)) {
        // Numeric columns get both a dimension AND aggregate measures
        dimensions[safeName] = {
          sql: `"${col.column_name}"`,
          type: "number",
          title: col.column_name,
        };
        measures[`avg_${safeName}`] = {
          sql: `"${col.column_name}"`,
          type: "avg",
          title: `Average ${col.column_name}`,
        };
        measures[`sum_${safeName}`] = {
          sql: `"${col.column_name}"`,
          type: "sum",
          title: `Sum ${col.column_name}`,
        };
        measures[`min_${safeName}`] = {
          sql: `"${col.column_name}"`,
          type: "min",
          title: `Min ${col.column_name}`,
        };
        measures[`max_${safeName}`] = {
          sql: `"${col.column_name}"`,
          type: "max",
          title: `Max ${col.column_name}`,
        };
      } else {
        dimensions[safeName] = {
          sql: `"${col.column_name}"`,
          type: "string",
          title: col.column_name,
        };
      }
    }

    // 4. Auto-detect forecast segments
    const segments = {};
    const hasForecastFlag = columns.some(c => c.column_name === "is_forecast");
    if (hasForecastFlag) {
      segments.historicalOnly = { sql: `"is_forecast" = false` };
      segments.forecastOnly = { sql: `"is_forecast" = true` };
    }

    // 5. Create the Cube model dynamically
    const cubeName = table_name
      .split("_")
      .map(w => w.charAt(0).toUpperCase() + w.slice(1))
      .join("");

    cube(cubeName, {
      sql: `SELECT * FROM public."${table_name}"`,
      title: table_name,
      measures,
      dimensions,
      segments,
    });
  }
});
