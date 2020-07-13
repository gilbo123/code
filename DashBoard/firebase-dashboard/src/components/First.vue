<template>
    <div class="demo">
    <h1>{{ title }}</h1>
    <h2>{{ msg }}</h2>
    </div>
</template>

<script>
export default {
    name: 'First',
    props: {
    title: String,
    msg: String
    }
}
</script>

<style scoped>
.demo {
	width: 500px;
  margin: auto;
  text-align: center;
  /* background-color: cyan; */
}
</style>
